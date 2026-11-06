# Showing example usage of the trained model. It should load the saved model weights, 
# run inference on test cases, print out any results, and provide visualisations where
# applicable. 

# Note: NumPy is allowed in this script for visualisation or loading data

#♥=========== Imports ===========♥#
from modules import build_model
from train import NUM_CLASSES

import numpy as np
import torch
#♥=========== Configuration ===========♥#

# run name (the name train.py gives its output files), (architecture, context)
RUNS = {
    "unet2d":    ("unet", "none"),
    "can_none":  ("can", "none"),
    "can_basic": ("can", "basic"),
    "can_large": ("can", "large"),
}
SHORT_NAMES = ["BG", "CSF", "GM", "WM"]
TISSUE = list(range(1, NUM_CLASSES)) # Every class except background

# Analysis calibration
BOUNDARY_RADIUS = 2 # This is a boundary band, marking for pixels within 2px of a ground truth boundary
MIN_CLASS_PIXELS = 30 # A class only counts on a slice if it covers at least this many pixels
MIN_BAND_PIXELS = 10 # Same rule but for the boundary band
MIN_SLICE_FG = 500 # Slices with less tissue than this are ignored when picking cases
MIN_COMPONENT_PIXELS = 5 # Connected componenets smaller than this are considered noise
GOOD_DICE = 0.90 # Target dice score as outlined in the brief
EDGE_FRACTION = 0.15 # Outer 15% of slice indices at each end count as "edge" slices
BIN_WIDTH = 8 # Slice-index bin width for the position plot
MIN_CASE_GAP = 6 # Failure cases from the saem subject must be at least this many slices apart

# Colours: labels (background, CSF, GM, WM) and the three error types (numpy used for matplotlib)
LABEL_COLOURS = np.array([[0.0, 0.0, 0.0], [0.56, 0.42, 0.75], [0.31, 0.65, 0.42], [0.91, 0.91, 0.91]])
ERROR_COLOURS = {
    "False positive (over-segmentation)": (0.90, 0.10, 0.10),
    "False negative (under-segmentation)": (0.10, 0.50, 1.00),
    "Tissue confusion (wrong tissue)": (1.00, 0.85, 0.00),
}

#♥=========== Inference ===========♥#
def load_trained_model(run_name, weights_dir, device):
    arch, context = RUNS[run_name]
    path = weights_dir / f"best_{run_name}.pth"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint at {path}. Train it first, or point --weights-dir at its folder.")

    model = build_model(arch, num_classes=NUM_CLASSES, context=context).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

@torch.no_grad()
def predict_test_set(model, loader, device):
    images, truths, predictions = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        predictions.append(logits.argmax(dim=1).cpu().to(torch.uint8))
        truths.append(y.cpu().to(torch.uint8))
        images.append((x[:, 0].cpu() * 255).round().to(torch.uint8))
    return torch.cat(images), torch.cat(truths), torch.cat(predictions)

#♥=========== Metrics for evaluation ===========♥#
def per_slice_confusion(preds, gts, num_classes, weights=None, chunk=64):
    assert gts.max() < num_classes and preds.max() < num_classes, "label outside 0..num_classes-1"
    n = len(gts)
    out = torch.zeros((n, num_classes, num_classes), dtype=torch.int64)
    for start in range(0, n, chunk):
        g = gts[start:start + chunk].to(torch.int64).reshape(-1, gts.shape[1] * gts.shape[2])
        p = preds[start:start + chunk].to(torch.int64).reshape(g.shape)
        offset = torch.arange(len(g))[:, None] * num_classes ** 2
        idx = (offset + g * num_classes + p).reshape(-1)
        w = None if weights is None else weights[start:start + chunk].reshape(-1).to(torch.float64)
        counts = torch.bincount(idx, weights=w, minlength=len(g) * num_classes ** 2)
        out[start:start + len(g)] = (
            torch.round(counts.to(torch.float64)).to(torch.int64).reshape(len(g),num_classes, num_classes))
    return out

def tp_fp_fn(conf):
    tp = torch.diagonal(conf, dim1=-2, dim2=-1)
    fn = conf.sum(dim=-1) - tp # ground truth is class c, predicted something else
    fp = conf.sum(dim=-2) - tp # predicted class c, ground truth is something else
    return tp, fp, fn

def dice_iou(tp, fp, fn, eps=1e-7):
    tp, fp, fn = tp.to(torch.float64), fp.to(torch.float64), fn.to(torch.float64)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou

def per_slice_dice(conf, min_pixels):
    tp, fp, fn = tp_fp_fn(conf)
    gt_area, pred_area = tp + fn, tp + fp
    dice = 2.0 * tp.to(torch.float64) / torch.clamp(gt_area + pred_area, min=1)
    return torch.where((gt_area >= min_pixels) | (pred_area >= min_pixels), dice, torch.full_like(dice, float("nan")))

def nan_mean(values, dim):
    valid = ~torch.isnan(values)
    total = torch.where(valid, values, torch.zeros_like(values)).sum(dim=dim)
    count = valid.sum(dim=dim)
    return torch.where(count > 0, total / torch.clamp(count, min=1), torch.full_like(total, float("nan")))

def boundary_band(gts, radius):
    edge = torch.zeros(gts.shape, dtype=torch.bool)
    vertical = gts[:, 1:, :] != gts[:, :-1, :]
    edge[:, 1:, :] |= vertical
    edge[:, :-1, :] |= vertical
    horizontal = gts[:, :, 1:] != gts[:, :, :-1]
    edge[:, :, 1:] |= horizontal
    edge[:, :, :-1] |= horizontal

    band = edge
    for _ in range(radius): # Grow by one pixel (4-neighbourhood) per step
        grown = band.clone()
        grown[:, 1:, :] |= band[:, :-1, :]
        grown[:, :-1, :] |= band[:, 1:, :]
        grown[:, :, 1:] |= band[:, :, :-1]
        grown[:, :, :-1] |= band[:, :, 1:]
        band = grown
    return band

def component_areas(mask):
    h, w = mask.shape
    stride = w + 2 # One zero column on each side so that runs never wrap onto the next row
    padded = torch.zeros((h, stride), dtype=torch.int8)
    padded[:, 1:-1] = mask
    change = torch.diff(padded.reshape(-1))
    starts = torch.nonzero(change == 1).flatten() + 1 # First pixel of each run
    ends = torch.nonzero(change == -1).flatten() + 1  # First pixel after each run
    if starts.numel() == 0:
        return torch.zeros(0, dtype=torch.int64)

    rows = torch.div(starts, stride, rounding_mode="floor")
    first_run_of_row = torch.searchsorted(rows, torch.arange(h + 1)).tolist()
    s = (starts % stride).tolist()
    e = (ends % stride).tolist()

    parent = list(range(len(s)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    # Runs in neighbouring rows that overlap in x belong to the same component
    for r in range(h - 1):
        i, i_end = first_run_of_row[r], first_run_of_row[r + 1]
        j, j_end = first_run_of_row[r + 1], first_run_of_row[r + 2]
        while i < i_end and j < j_end:
            if s[i] < e[j] and s[j] < e[i]:
                root_i, root_j = find(i), find(j)
                if root_i != root_j:
                    parent[root_j] = root_i
            if e[i] < e[j]:
                i += 1
            else:
                j += 1

    lengths = torch.tensor(e, dtype=torch.float64) - torch.tensor(s, dtype=torch.float64)
    roots = torch.tensor([find(i) for i in range(len(s))], dtype=torch.int64)
    areas = torch.bincount(roots, weights=lengths)
    return areas[areas > 0].to(torch.int64)

def component_counts(label_maps, classes, min_pixels):
    counts = torch.zeros((len(label_maps), len(classes)), dtype=torch.int32)
    for n, labels in enumerate(label_maps):
        for k, c in enumerate(classes):
            mask = labels == c
            if mask.any():
                counts[n, k] = (component_areas(mask) >= min_pixels).sum()
    return counts

def analyse_model(preds, gts, band, gt_components):
    conf = per_slice_confusion(preds, gts, NUM_CLASSES)
    conf_band = per_slice_confusion(preds, gts, NUM_CLASSES, weights=band)

    dice, iou = dice_iou(*tp_fp_fn(conf.sum(dim=0)))
    band_dice, _ = dice_iou(*tp_fp_fn(conf_band.sum(dim=0)))
    slice_dice = per_slice_dice(conf, MIN_CLASS_PIXELS)
    slice_band = per_slice_dice(conf_band, MIN_BAND_PIXELS)

    pred_components = component_counts(preds, TISSUE, MIN_COMPONENT_PIXELS)
    # Extra fragments the model invented. A class with MORE pieces in the ground truth is not penalised
    excess = torch.clamp(pred_components - gt_components, min=0).sum(dim=1)

    fg_dice = slice_dice[:, 1:]
    worst_class = torch.amin(torch.where(torch.isnan(fg_dice), torch.ones_like(fg_dice), fg_dice), dim=1)
    return {
        "dice": dice, "iou": iou, "band_dice": band_dice,
        "slice_dice": slice_dice,
        "slice_dice_mean": nan_mean(slice_dice[:, 1:], dim=1),
        "slice_band_mean": nan_mean(slice_band[:, 1:], dim=1),
        "excess": excess,
        "usable": worst_class >= GOOD_DICE, # every tissue class on this slice reaches the target
    }


def main():
    pass

if __name__ == "__main__":
    main()