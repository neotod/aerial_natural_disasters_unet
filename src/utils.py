import math
import os
import json
import shutil
import torch


def convert_number_to_4_digits_str(num):
    leading_zeros = ["0" for _ in range(4 - len(str(num)))]
    leading_zeros += list(str(num))

    return "".join(leading_zeros)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_model_checkpoint_best_only(
    args, epoch, model, metric_val, learning_rate, criteria="min"
):
    # model save checkpoint
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"best_checkpoint")

    try:
        if os.path.exists(checkpoint_dir):
            with open(os.path.join(checkpoint_dir, "configs.json"), "r") as f:
                configs = json.load(f)
        else:
            configs = {
                "encoder": args.encoder,
                "metric": math.inf if criteria == "min" else -math.inf,
                "lr": learning_rate,
                "epoch": epoch,
            }

        if (criteria == "min" and metric_val < configs["metric"]) or (
            criteria == "max" and metric_val > configs["metric"]
        ):
            shutil.rmtree(checkpoint_dir, ignore_errors=True)
            os.makedirs(checkpoint_dir)
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
            with open(os.path.join(checkpoint_dir, "configs.json"), "w") as f:
                json.dump(
                    {
                        "encoder": args.encoder,
                        "metric": round(metric_val, 7),
                        "lr": learning_rate,
                        "epoch": epoch,
                    },
                    f,
                )

            print(f"model .pth and smp configs at epoch {epoch} saved!\n")
            return True

    except Exception as e:
        print(f"Skipping best checkpoint saving because '{e}'\n")
        return False


def save_model_checkpoint(args, epoch, learning_rate, model):
    # model save checkpoint
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"checkpoint_{epoch}")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.makedirs(checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
    with open(os.path.join(checkpoint_dir, "configs.json"), "w") as f:
        json.dump(
            {
                "encoder": args.encoder,
                "metric": None,
                "lr": learning_rate,
                "epoch": epoch,
            },
            f,
        )

    print(f"model .pth and smp configs at epoch {epoch} saved!")
    print("")

    return True
