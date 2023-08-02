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
    parser.add_argument("--lr", "-l", type=float, default=0.1)
    parser.add_argument("--base_lr", type=float, default=5e-3)
    parser.add_argument("--max_lr", type=float, default=25e-3)
    parser.add_argument("--lr_sched", type=str, default="reduce_on_plateau")
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
        "--loss_fn", type=str, default="ce", help="focal | ce | dice | ce+dice"
    )
    parser.add_argument("--save_best_only", type=bool, default=True)
    parser.add_argument("--resume_from_best", type=bool, default=False)

    return parser.parse_args()


def get_model(args):
    if args.resume_from_best:
        dir = os.path.join(args.checkpoints_dir, f"best_checkpoint")

        model_path = os.path.join(dir, "model.pth")
        configs_path = os.path.join(
            dir,
            "configs.json",
        )

        try:
            with open(configs_path, "r") as f:
                configs = json.load(f)

            if configs["encoder"] != args.encoder:
                raise Exception(
                    f"model checkpoint's encoder ({configs['encoder']}) is different from the wanted encoder ({args.encoder})"
                )

        except:
            raise Exception(f"can't fine the loaded model configs at {configs_path}")

        model = smp.Unet(
            encoder_name=configs["encoder"],
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path))
            print(f"model loaded from save checkpoint at epoch {configs['epoch']}")
        else:
            raise Exception(
                f"can't fine the loaded model checkpoint at epoch {configs['epoch']} at {model_path}"
            )

        return model, configs

    else:
        model = smp.Unet(
            encoder_name=args.encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=14,
        )

        return model, {
            "epoch": 1,
            "metric": None,
            "lr": args.lr,
            "encoder": args.encoder,
        }


def do_epoch_train(model: nn.Module, train_dl, loss_fn, opt, lr_sched, logger=None):
    train_loss = count = 0
    model.train()

    train_dl.dataset.dataset.phase = "train"
    lr_before = utils.get_lr(opt)
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

        if logger and os.getenv("WANDB_LOG_IMAGES") in ["true", "True"]:
            logger.log_images(x, y, y_pred)

        if args.lr_sched == "cyclic":
            lr_sched.step()

    train_loss /= count
    lr_after = utils.get_lr(opt)

    return train_loss, lr_before, lr_after


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

    model, configs = get_model(args)

    opt = torch.optim.Adam(model.parameters(), lr=configs["lr"])

    if args.lr_sched == "reduce_on_plateau":
        lr_sched = lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=2)
    elif args.lr_sched == "cyclic":
        lr_sched = lr_scheduler.CyclicLR(
            opt,
            args.base_lr,
            args.max_lr,
            step_size_up=len(train_dl) * 5,
            mode="triangular",
            cycle_momentum=False,
        )
    else:
        raise Exception(
            'lr_sched is not valid. It should be either "reduce_on_plateau" or "cyclic"'
        )

    # freezing encoder's weights
    for param in model.encoder.parameters():
        param.requires_grad = False

    model.to(device)

    epoch = configs["epoch"]  # it's either saved epoch or 1 (fresh model)
    for _ in range(args.epochs):
        t1 = time.time()

        train_loss, lr_before, lr_after = do_epoch_train(
            model, train_dl, loss_fn, opt, lr_sched, logger=logger
        )
        val_loss, metrics = do_epoch_val(model, val_dl, loss_fn)

        if args.lr_sched == "reduce_on_plateau":
            lr_before = utils.get_lr(opt)
            lr_sched.step(val_loss)
            lr_after = utils.get_lr(opt)

        if lr_before != lr_after:
            print(f"learning rate changed from {lr_before} to {lr_after}")

        if args.save_best_only:
            utils.save_model_checkpoint_best_only(
                args, epoch, model, metrics["iou"], lr_after, criteria="max"
            )
        else:
            utils.save_model_checkpoint(args, epoch, lr_after, model, criteria="max")

        print(
            "epoch: {epoch} | train_loss: {loss:.6f} | val_loss: {val_loss:.6f} | iou: {iou:.6f} | time: {time:.3f}s\n".format(
                epoch=epoch,
                loss=train_loss,
                time=time.time() - t1,
                val_loss=val_loss,
                iou=metrics["iou"],
            )
        )

        if logger:
            logger.log_epoch(train_loss, val_loss, metrics, epoch, lr_after)

        epoch += 1


if __name__ == "__main__":
    args = get_args()

    print("configs:")
    print(json.dumps(args.__dict__, indent=3))

    main(args)
