import json
import os
import tqdm
import time
import argparse
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import segmentation_models_pytorch as smp
from evaluate import eval_model
from src import data_loader, utils
from src.wandb_logger import WandBLogger

load_dotenv()

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_dir",
        type=str,
        default="./data/",
        help="address to training and validation images directory",
    )
    parser.add_argument("--lr", "-l", type=float, required=True)
    parser.add_argument("--epochs", "-e", type=int, required=True)
    parser.add_argument(
        "--validation_split",
        "-v",
        type=float,
        default=0.2,
        help="percent of data for validation",
    )
    parser.add_argument("--batch_size", "-b", type=int)
    parser.add_argument("--encoder", type=str, default="resnet18")
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints/")
    parser.add_argument(
        "--continue_epoch",
        type=int,
        default=0,
        help="continue training the model by loading the saved model at this epoch and not from scratch",
    )
    parser.add_argument(
        "--loss_fn", type=str, default="ce", help="focal | ce | dice | ce+dice"
    )

    return parser.parse_args()


def get_model(args):
    if args.continue_epoch != 0:
        model_path = os.path.join(
            args.checkpoints_dir, f"checkpoint_{args.continue_epoch}", "model.pth"
        )
        smp_configs_path = os.path.join(
            args.checkpoints_dir,
            f"checkpoint_{args.continue_epoch}",
            "smp_configs.json",
        )

        with open(smp_configs_path, "r") as f:
            smp_configs = json.load(f)

        if smp_configs["encoder"] != args.encoder:
            raise Exception(
                f"model checkpoint's encoder ({smp_configs['encoder']}) is different from the wanted encoder ({args.encoder})"
            )

        model = smp.Unet(
            encoder_name=smp_configs["encoder"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"model loaded from save checkpoint at epoch {args.continue_epoch}")
        else:
            raise Exception(
                f"can't fine the loaded model checkpoint at epoch {args.continue_epoch} at {model_path}"
            )

    else:
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )

    return model


def do_epoch_train(model: nn.Module, train_dl, loss_fn, opt, logger=None):
    train_loss = count = 0
    model.train()

    train_dl.dataset.dataset.phase = "train"
    for x, y in tqdm.tqdm(train_dl):
        x = x.to(device)

        y_pred = model(x)
        y = y.to(device)

        opt.zero_grad()

        if args.loss_fn in ["ce+dice"]:
            loss_i = loss_fn["ce"](y_pred, y)
            loss_i += loss_fn["dice"](y_pred, y)
        else:
            loss_i = loss_fn(y_pred, y)

        loss_i.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        train_loss += loss_i.item()
        count += 1

        if logger and os.getenv('WANDB_LOG_IMAGES') in ['true', 'True']:
            print("loggingg images")
            logger.log_images(x, y, y_pred)

    train_loss /= count

    return train_loss


def do_epoch_val(model: nn.Module, val_dl, loss_fn):
    metrics = {
        "iou": 0,
    }
    val_loss = count = 0

    val_dl.dataset.dataset.phase = "val"
    model.eval()
    with torch.no_grad():
        for x, y in val_dl:
            x = x.to(device)

            y_pred = model(x)
            y = y.to(device)

            if args.loss_fn in ["ce+dice"]:
                val_loss_i = loss_fn["ce"](y_pred, y)
                val_loss_i += loss_fn["ce"](y_pred, y)
            else:
                val_loss_i = loss_fn(y_pred, y)

            val_loss += val_loss_i.item()
            count += 1

            metrics_i = eval_model(model, x, y)

            metrics["iou"] += metrics_i["iou"]

        val_loss /= count
        metrics["iou"] /= count

    return val_loss, metrics


def main(args):
    # (Initialize logging)
    logger = None
    if args.debug:
        logger = WandBLogger(args)

    if args.loss_fn == "focal":
        loss_fn = smp.losses.FocalLoss(mode="multiclass")
    elif args.loss_fn == "dice":
        loss_fn = smp.losses.DiceLoss(mode="multiclass")
    elif args.loss_fn == "ce":
        loss_fn = nn.CrossEntropyLoss()
    elif args.loss_fn == "ce+dice":
        loss_fn = {
            "ce": nn.CrossEntropyLoss(),
            "dice": smp.losses.DiceLoss(mode="multiclass"),
        }
    else:
        raise Exception("Loss is not valid.")

    train_dl, val_dl = data_loader.get_train_data_loaders(
        args.train_dir, args.validation_split, args.batch_size
    )

    model = get_model(args)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2)

    # freezing encoder's weights
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.to(device)

    epoch = args.continue_epoch + 1 if args.continue_epoch != 0 else 1
    for _ in range(args.epochs):
        t1 = time.time()

        train_loss = do_epoch_train(model, train_dl, loss_fn, opt, logger=logger)
        val_loss, metrics = do_epoch_val(model, val_dl, loss_fn)

        lr_before = utils.get_lr(opt)
        lr_sched.step(val_loss)
        lr_after = utils.get_lr(opt)

        if lr_before != lr_after:
            print(f"learning rate changed from {lr_before} to {lr_after}")

        print(
            "epoch: {epoch} | train_loss: {loss:.6f} | val_loss: {val_loss:.6f} | iou: {iou:.6f} | time: {time:.3f}s".format(
                epoch=epoch,
                loss=train_loss,
                time=time.time() - t1,
                val_loss=val_loss,
                iou=metrics["iou"],
            )
        )

        if logger:
            logger.log_epoch(train_loss, val_loss, metrics, epoch, lr_after)

        utils.save_model_checkpoint(args, epoch, model)
        epoch += 1


if __name__ == "__main__":
    args = get_args()

    print("configs:")
    print(json.dumps(args.__dict__, indent=3))

    main(args)
