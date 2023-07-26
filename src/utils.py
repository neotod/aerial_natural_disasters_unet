import os
import json
import shutil
import torch
import segmentation_models_pytorch as smp


def convert_number_to_4_digits_str(num):
    leading_zeros = ["0" for _ in range(4 - len(str(num)))]
    leading_zeros += list(str(num))

    return "".join(leading_zeros)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def save_model_checkpoint(args, epoch, model):
    # model save checkpoint
    checkpoint_dir = os.path.join(args.checkpoints_dir, f"checkpoint_{epoch}")
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.makedirs(checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pth"))
    with open(os.path.join(checkpoint_dir, "smp_configs.json"), "w") as f:
        json.dump({"encoder": args.encoder}, f)

    print(f"model .pth and smp configs at epoch {epoch} saved!")
    print("")
