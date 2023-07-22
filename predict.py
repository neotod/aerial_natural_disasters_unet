from PIL import Image
import numpy as np
import json
import os
import argparse

import torch
from torchvision import transforms
import segmentation_models_pytorch as smp

from src import utils

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", "-m", required=True, type=str, help="address to model .pth file"
    )
    parser.add_argument(
        "--smp_configs_path",
        "-s",
        required=True,
        type=str,
        help="address smp configs json file",
    )
    parser.add_argument("--input_dir", type=str, help="address to images to predict")
    parser.add_argument(
        "--output_dir", type=str, help="address to save predicted masks"
    )

    return parser.parse_args()


args = get_args()

print("configs:")
print(json.dumps(args.__dict__, indent=3))

files = os.listdir(args)
x_transforms = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.46077183, 0.45584197, 0.41929824],
            std=[0.18551224, 0.17078055, 0.17699541],
        ),
    ]
)

with open(os.path.join(args.smp_configs_path)) as f:
    model_configs = json.load(f)

model = smp.Unet(
    encoder_name=model_configs["encoder"],
    encoder_weights="imagenet",
    in_channels=3,
    classes=14,
)
model.load_state_dict(torch.load(args.model_path))

model.to(device)

model.eval()
with torch.no_grad():
    preds = []

    for img_name in sorted(os.listdir(args.input_dir)):
        img_path = os.path.join(args.input_dir, img_name)
        img = Image.open(img_path)
        img = x_transforms(img)
        img = img.unsqueeze(0)

        img = img.to(device)

        y_pred = model(img)

        preds.append(y_pred.squeeze().permute(1, 2, 0).cpu().detach().numpy())

    preds = np.array(preds)

for i, mask_pred in enumerate(preds):
    print(f"saving predicted mask #{i}")

    mask_pred = np.argmax(mask_pred, axis=2)

    mask_path = os.path.join(
        args.output_dir, f"{utils.convert_number_to_4_digits_str(i)}.png"
    )

    img = Image.fromarray(mask_pred.astype(np.uint8))
    img.save(mask_path)
