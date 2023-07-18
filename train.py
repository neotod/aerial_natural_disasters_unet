import shutil
import json
import os
import tqdm
import time
import argparse

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from evaluate import eval_model
from src import data_loader

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, default='./data/', help='address to training and validation images directory')
    parser.add_argument('--lr', '-l', type=float)
    parser.add_argument('--epochs', '-e', type=int)
    parser.add_argument('--validation_split', '-v', type=float, default=0.2, help='percent of data for validation')
    parser.add_argument('--batch_size', '-b', type=int)
    parser.add_argument('--encoder', type=str, default='resnet18')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/')
    parser.add_argument('--continue_epoch', type=int, default=0, help='continue training the model by loading the saved model at this epoch and not from scratch')
    parser.add_argument('--loss', type=str, default='ce', help='focal | ce | dice')


    return parser.parse_args()


args = get_args()

print('configs:')
print(json.dumps(args.__dict__, indent=3))


if args.loss == 'focal':
    loss_ = smp.losses.FocalLoss(mode='multiclass', gamma=3/4)
elif args.loss == 'dice':
    loss_ = smp.losses.DiceLoss(mode='multiclass')
elif args.loss == 'ce':
    loss_ = nn.CrossEntropyLoss()
else:
    raise Exception('Loss is not valid.')


train_dl, val_dl = data_loader.get_train_data_loaders(args.train_dir, args.validation_split, args.batch_size)

if args.continue_epoch != 0:
    model_path = os.path.join(args.checkpoints_dir, f'checkpoint_{args.continue_epoch}', 'model.pth')
    smp_configs_path = os.path.join(args.checkpoints_dir, f'checkpoint_{args.continue_epoch}', 'smp_configs.json')

    with open(smp_configs_path, 'r') as f:
        smp_configs = json.load(f)

    model = smp.Unet(
        encoder_name=smp_configs['encoder'],
        encoder_weights='imagenet',
        in_channels=3,
        classes=14
    )

    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(
                model_path
            )
        )
        print(f'model loaded from save checkpoint at epoch {args.continue_epoch}')
    else:
        print(f"can't fine the loaded model checkpoint at epoch {args.continue_epoch} at {model_path}")

else:
    model = smp.Unet(
        encoder_name=args.encoder,
        encoder_weights='imagenet',
        in_channels=3,
        classes=14
    )
    
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
# freezing encoder's weights
for param in model.encoder.parameters():
    param.requires_grad = False

model.to(device)

cnt = 0
for ep in range(args.epochs):
    train_loss = 0
    t1 = time.time()

    model.train()

    train_dl.dataset.dataset.phase = 'train'
    for (x,y) in tqdm.tqdm(train_dl):
        x = x.to(device)

        y_pred = model(x)
        y = y.to(device)

        opt.zero_grad()
        loss_i = loss_(y_pred, y)

        loss_i.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        train_loss += loss_i.item()
        cnt += 1

    train_loss /= cnt

    val_loss = cnt =0
    val_dl.dataset.dataset.phase = 'val'
    model.eval()
    with torch.no_grad():
        for (x,y) in val_dl:
            x = x.to(device)

            y_pred = model(x)
            y = y.to(device)

            val_loss = loss_(y_pred, y).item()
            cnt += 1

            metrics = eval_model(model, x,y)

        val_loss /= cnt
        metrics['iou'] /= cnt
        metrics['accuracy'] /= cnt

    print('epoch: {epoch} | train_loss: {loss:.6f} | val_loss: {val_loss:.6f} | iou: {iou} | accuracy: {acc} | time: {time:.3f}s'.format(epoch=ep, loss=loss_i, time=time.time() - t1, val_loss=val_loss, iou=metrics['iou'], acc=metrics['accuracy']))

    #uracy save checkpoint
    checkpoint_dir = os.path.join(args.checkpoints_dir, f'checkpoint_{ep}')
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    os.mkdir(checkpoint_dir)
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'model.pth'))
    with open(os.path.join(checkpoint_dir, 'smp_configs.json'), 'w') as f:
        json.dump({
            "encoder": args.encoder
        }, f)

    print(f'model .pth and smp configs at epoch {ep} saved!')
    print()
    