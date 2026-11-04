# Helper file utils.py for organisation ♥

#♥=========== Imports ===========♥#
import torch
import torch.nn as nn
import torch.nn.functional as F

#♥=========== Loss ===========♥# | || || |_
class DiceLoss(nn.Module):
    """Soft multi-class Dice loss, averaged over classes"""

    def __init__(self, num_classes, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits, targets):
        # Logits: (B, C, H, W), targets: (B, H, W) integer class labels
        probs = F.softmax(logits, dim=1)
        onehot = F.one_hot(targets, self.num_classes).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3) # Sum over batch and space, keep class dimension
        intersection = (probs * onehot).sum(dims)
        denominator = probs.sum(dims) + onehot.sum(dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        return 1.0 - dice.mean()

class CombinedLoss(nn.Module):
    """Cross-entropy + Dice. CE gives stable gradients, Dice handles the class imbalance"""

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(num_classes)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        return self.ce_weight * self.ce(logits, targets) + self.dice_weight * self.dice(logits, targets)

#♥=========== Metrics ===========♥#
@torch.no_grad()
def confusion_stats(logits, targets, num_classes):
    """
    Per-class true positives, false positives and false negatives for one batch.
    Accumulates these over a whole dataset, then call dice_iou_from_stats once:
    This is to avoid Nan and distortion from slices where a class is absent
    :param logits:
    :param targets:
    :param num_classes:
    :return: tp, fp, fn - each a float64 CPU tensor of shape (num_classes)
    """
    preds = logits.argmax(dim=1)
    tp, fp, fn = [], [], []
    for c in range(num_classes):
        p = preds == c
        t = targets == c
        tp.append((p & t).sum())
        fp.append((p & ~t).sum())
        fn.append((~p & t).sum())

    to_cpu = lambda values: torch.stack(values).double().cpu()
    return to_cpu(tp), to_cpu(fp), to_cpu(fn)

def dice_iou_from_stats(tp, fp, fn, eps=1e-7):
    """ Per class Dice (DSC) and IoU from accumulated tp/fp/fn"""
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou