import numpy as np
import torch
import segmentation_models_pytorch as smp

device = ('cuda' if torch.cuda.is_available() else 'cpu')

def eval_model(model, x, y):
    metrics = {
        'iou': 0,
    }

    x = x.to(device)

    y_pred = model(x)
    y_pred = torch.argmax(y_pred.permute(0, 2,3,1).cpu(), dim=3)

    tp, fp, fn, tn = smp.metrics.get_stats(y_pred, y.long().cpu(), mode='multiclass', num_classes=14)
    metrics['iou'] = smp.metrics.iou_score(tp, fp, fn, tn, reduction='micro').item()

    return metrics

        