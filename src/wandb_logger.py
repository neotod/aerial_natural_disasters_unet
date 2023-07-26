import os
import wandb
from src import const


class WandBLogger:
    def __init__(self, args):
        self.run_id = os.getenv(
            "WANDB_RUN_ID",
            "u-net_{encoder}_{loss}_{lr}".format(
                encoder=args.encoder, loss=args.loss_fn, lr=args.lr
            ),
        )

        self.experiment = wandb.init(
            id=self.run_id,
            project=os.getenv("WANDB_PROJECT_NAME"),
            resume="allow",
            anonymous="allow",
            config={
                "architecture": "u-net",
                "encoder": args.encoder,
                "loss": args.loss_fn,
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        )

    def log_epoch(self, train_loss, val_loss, metrics, epoch, lr):
        self.experiment.log(
            {
                "train loss": train_loss,
                "validation loss": val_loss,
                "iou": metrics["iou"],
                "epoch": epoch,
                "learning rate": lr,
            }
        )

    def log_images(self, x, y, y_pred):
        for x_i, y_i, y_pred_i in zip(x, y, y_pred):
            self.experiment.log(
                {
                    "images": wandb.Image(
                        x_i.cpu().permute(1, 2, 0).numpy(),
                        masks={
                            "predications": {
                                "mask_data": y_pred_i.argmax(dim=1)
                                .float()
                                .cpu()
                                .numpy(),
                                "class_labels": const.CLASS_NAMES,
                            },
                            "ground_truth": {
                                "mask_data": y_i.float().cpu().numpy(),
                                "class_labels": const.CLASS_NAMES,
                            },
                        },
                    ),
                }
            )
