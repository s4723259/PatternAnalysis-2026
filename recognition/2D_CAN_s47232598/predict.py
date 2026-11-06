# Showing example usage of the trained model. It should load the saved model weights, 
# run inference on test cases, print out any results, and provide visualisations where
# applicable. 

# Note: NumPy is allowed in this script for visualisation or loading data

# Usage:
#   python predict.py                                      # 2D U-Net (baseline) vs CAN
#   python predict.py --models unet2d can_none can_large   # first = baseline, last = model under test
#   python predict.py --show                               # also open the figures in windows

#♥=========== Imports ===========♥#
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from modules import build_model
from train import load_data, get_device, NUM_CLASSES, CLASS_NAMES

#♥=========== Configuration ===========♥#
HERE = Path(__file__).resolve().parent
# run name (the name train.py gives its output files), (architecture, context)
RUNS = {
    "unet2d":    ("unet", "none"),
    "can_none":  ("can", "none"),
    "can_basic": ("can", "basic"),
    "can_large": ("can", "large"),
}
DISPLAY_NAMES = {
    "unet2d":    "2D U-Net",
    "can_none":  "CAN front end only",
    "can_basic": "CAN (basic context)",
    "can_large": "CAN (large context)",
}
SHORT_NAMES = ["BG", "CSF", "GM", "WM"]
TISSUE = list(range(1, NUM_CLASSES)) # Every class except background

# Analysis calibration
BOUNDARY_RADIUS = 2         # This is a boundary band, marking for pixels within 2px of a ground truth boundary
MIN_CLASS_PIXELS = 30       # A class only counts on a slice if it covers at least this many pixels
MIN_BAND_PIXELS = 10        # Same rule but for the boundary band
MIN_SLICE_FG = 500          # Slices with less tissue than this are ignored when picking cases
MIN_COMPONENT_PIXELS = 5    # Connected componenets smaller than this are considered noise
GOOD_DICE = 0.90            # Target dice score as outlined in the brief
EDGE_FRACTION = 0.15        # Outer 15% of slice indices at each end count as "edge" slices
BIN_WIDTH = 8               # Slice-index bin width for the position plot
MIN_CASE_GAP = 6            # Failure cases from the saem subject must be at least this many slices apart

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
    """
    Runs the model over a loader and returns CPU tensors in the loader's order:
        images (N, H, W) uint8, ground truth (N, H, W) uint8, predictions (N, H, W) uint8
    """
    images, truths, predictions = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        predictions.append(logits.argmax(dim=1).cpu().to(torch.uint8))
        truths.append(y.cpu().to(torch.uint8))
        images.append((x[:, 0].cpu() * 255).round().to(torch.uint8))
    return torch.cat(images), torch.cat(truths), torch.cat(predictions)

#♥=========== Metrics for evaluation ===========♥#
def per_slice_confusion(preds, gts, num_classes, weights=None, chunk=64):
    """
    Confusion matrix of every slice at once.
    :param weights: optional boolean (N, H, W) tensor: only pixels where it is True are counted
    :return: (N, C, C) int64 tensor, entry [n, g, p] = pixels of slice n with ground truth g predicted as p
    """
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
    """Per-class true positives, false positives, false negatives from (..., C, C) confusion matrices"""
    tp = torch.diagonal(conf, dim1=-2, dim2=-1)
    fn = conf.sum(dim=-1) - tp # ground truth is class c, predicted something else
    fp = conf.sum(dim=-2) - tp # predicted class c, ground truth is something else
    return tp, fp, fn

def dice_iou(tp, fp, fn, eps=1e-7):
    """Same formulas as utils.dice_iou_from_stats, so the numbers are comparable with train.py"""
    tp, fp, fn = tp.to(torch.float64), fp.to(torch.float64), fn.to(torch.float64)
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    return dice, iou

def per_slice_dice(conf, min_pixels):
    """
    (N, C) Dice for every slice and class. NaN where the class is (almost) absent from both the ground truth
    and the prediction. A class that is absent from the ground truth but predicted anyway scores 0 (a failure).
    """
    tp, fp, fn = tp_fp_fn(conf)
    gt_area, pred_area = tp + fn, tp + fp
    dice = 2.0 * tp.to(torch.float64) / torch.clamp(gt_area + pred_area, min=1)
    return torch.where((gt_area >= min_pixels) | (pred_area >= min_pixels), dice, torch.full_like(dice, float("nan")))

def nan_mean(values, dim):
    """Mean that ignores NaN and returns NaN (no warning) when everything is NaN"""
    valid = ~torch.isnan(values)
    total = torch.where(valid, values, torch.zeros_like(values)).sum(dim=dim)
    count = valid.sum(dim=dim)
    return torch.where(count > 0, total / torch.clamp(count, min=1), torch.full_like(total, float("nan")))

def boundary_band(gts, radius):
    """
    Boolean (N, H, W): pixels within about `radius` px of a boundary between two ground-truth regions
    (including the tissue / background outline). Global Dice is dominated by region interiors, which are easy;
    Dice restricted to this band measures boundary precision.
    """
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
    """
    Areas of the 4-connected components of a 2D boolean mask. Union-find over the horizontal runs of
    pixels, so no SciPy is needed. Returns an int64 tensor with one area per component.
    """
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
    """(N, len(classes)) number of connected components (of at least min_pixels) per slice and class"""
    counts = torch.zeros((len(label_maps), len(classes)), dtype=torch.int32)
    for n, labels in enumerate(label_maps):
        for k, c in enumerate(classes):
            mask = labels == c
            if mask.any():
                counts[n, k] = (component_areas(mask) >= min_pixels).sum()
    return counts

def analyse_model(preds, gts, band, gt_components):
    """Everything the report needs about one model, as a dictionary of arrays"""
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

#♥=========== Choosing the failure cases ===========♥#
def pick_unused(score, chosen, subject, slice_idx):
    """Highest-scoring slice that is not (nearly) the same slice of the same subject as a chosen case"""
    for n in torch.argsort(-score).tolist(): # NaN sorts last, like in NumPy, so it is never reached before -inf
        if not torch.isfinite(score[n]):
            return None
        if all(not (subject[n] == subject[m] and
                    abs(int(slice_idx[n]) - int(slice_idx[m])) < MIN_CASE_GAP) for m in chosen):
            return n
    return None

def select_failure_cases(stats, runs, subject, slice_idx, edge, eligible):
    """
    Picks up to five slices, each for a different reason, so that they tell different stories
    (the five worst slices would usually be neighbouring slices of one subject).
    base = first model (the baseline), test = last model (the one being investigated).
    """
    base, test = runs[0], runs[-1]
    S = {r: stats[r]["slice_dice_mean"] for r in runs}
    ok = eligible & torch.isfinite(S[test])

    criteria = [
        ("A", "Worst boundary slice",
         f"lowest boundary-band Dice ({DISPLAY_NAMES[test]})", -stats[test]["slice_band_mean"], ok),
        ("B", "Worst edge slice",
         f"lowest Dice among the outer {EDGE_FRACTION:.0%} of slices at each end ({DISPLAY_NAMES[test]})",
         -S[test], ok & edge),
        ("C", "Spurious fragments",
         f"most extra disconnected pieces ({DISPLAY_NAMES[test]})",
         stats[test]["excess"] + (1 - torch.nan_to_num(S[test], nan=1.0)) * 1e-3, ok & (stats[test]["excess"] > 0)),
    ]
    if base != test:
        gain = S[test] - S[base]
        ok_both = ok & torch.isfinite(S[base])
        criteria += [
            ("D", "Context hurts",
             f"{DISPLAY_NAMES[test]} much worse than {DISPLAY_NAMES[base]}", -gain, ok_both & (gain < -0.01)),
            ("E", "Context helps",
             f"{DISPLAY_NAMES[test]} much better than {DISPLAY_NAMES[base]}", gain, ok_both & (gain > 0.01)),
        ]

    cases, chosen = [], []
    for tag, title, reason, score, allowed in criteria:
        n = pick_unused(torch.where(allowed, score, torch.full_like(score, float("-inf"))), chosen, subject, slice_idx)
        if n is None:
            print(f"(No slice qualifies for case {tag}: {title.lower()})")
            continue
        chosen.append(n)
        cases.append({"tag": tag, "title": title, "reason": reason, "index": n})
    return cases

#♥=========== Console reports ===========♥#
# Please note, this code was generated by AI
def fmt(value, spec):
    return "-" if value is None else format(value, spec)

def print_metric_table(title, runs, values):
    print(f"\n{title}")
    print(f"{'Model':<22}" + "".join(f"{name:>14}" for name in CLASS_NAMES) + f"{'Mean (fg)':>12}")
    for run in runs:
        v = values[run]
        print(f"{DISPLAY_NAMES[run]:<22}" +
              "".join(f"{x:>14.4f}" for x in v.tolist()) + f"{v[1:].mean().item():>12.4f}")

def print_cost_table(runs, weights_dir):
    """Cost figures recorded by train.py (results_<run>.json)"""
    rows = []
    for run in runs:
        path = weights_dir / f"results_{run}.json"
        if path.exists():
            with open(path) as f:
                r = json.load(f)
            params = r["params_total"] / 1e6
            rows.append((DISPLAY_NAMES[run], fmt(params, ".2f"), fmt(r.get("gflops_per_slice"), ".1f"),
                         fmt(r.get("ms_per_slice"), ".2f"), fmt(r.get("peak_infer_mb"), ".0f"),
                         fmt(r.get("peak_train_mb"), ".0f"), fmt(r.get("train_minutes"), ".1f")))
    if not rows:
        return
    print("\nComputational cost (recorded by train.py)")
    header = ("Model", "Params (M)", "GFLOPs/slice", "ms/slice", "Infer MB", "Train MB", "Train min")
    print(f"{header[0]:<22}" + "".join(f"{h:>14}" for h in header[1:]))
    for row in rows:
        print(f"{row[0]:<22}" + "".join(f"{x:>14}" for x in row[1:]))

def check_against_training(run, dice, weights_dir):
    """The Dice recomputed here must match what train.py stored, otherwise the split or checkpoint is different"""
    path = weights_dir / f"results_{run}.json"
    if not path.exists():
        return
    with open(path) as f:
        saved = torch.tensor(json.load(f)["test_dice"], dtype=torch.float64)
    worst = (saved - dice).abs().max().item()
    verdict = "matches" if worst < 1e-3 else "DIFFERS: is this the same checkpoint and test split?"
    print(f"  {DISPLAY_NAMES[run]:<22} recomputed Dice {verdict} results_{run}.json (max difference {worst:.1e})")

def bin_by_slice(slice_idx):
    lo = int(slice_idx.min()) # Works for tensors and (for the plot) NumPy arrays alike
    bin_id = (slice_idx - lo) // BIN_WIDTH
    return lo, bin_id, int(bin_id.max()) + 1

def print_usability_summary(runs, stats, slice_idx, edge, eligible):
    """Numbers behind the recommendation: how often is a slice good enough to contour automatically?"""
    print(f"\nUsability: a slice is 'usable' when EVERY tissue class reaches Dice >= {GOOD_DICE} "
          f"(slices with >= {MIN_SLICE_FG} tissue pixels)")
    print(f"{'Model':<22}{'Slices':>10}{'Usable':>10}{'Usable (central)':>20}{'Usable (edge)':>16}"
          f"{'Mean Dice (central)':>22}{'Mean Dice (edge)':>19}")
    central = eligible & ~edge
    edge_e = eligible & edge
    for run in runs:
        s = stats[run]
        usable = s["usable"].to(torch.float64)
        print(f"{DISPLAY_NAMES[run]:<22}{int(eligible.sum()):>10}{usable[eligible].mean().item():>10.1%}"
              f"{usable[central].mean().item():>20.1%}{usable[edge_e].mean().item():>16.1%}"
              f"{torch.nanmean(s['slice_dice_mean'][central]).item():>22.4f}"
              f"{torch.nanmean(s['slice_dice_mean'][edge_e]).item():>19.4f}")

    print("\nSpurious fragments (extra disconnected pieces that do not exist in the ground truth)")
    for run in runs:
        excess = stats[run]["excess"][eligible]
        print(f"  {DISPLAY_NAMES[run]:<22} slices with >= 1 extra fragment: "
              f"{(excess > 0).to(torch.float64).mean().item():>6.1%}   "
              f"mean extra fragments per slice: {excess.to(torch.float64).mean().item():.3f}")

    lo, bin_id, n_bins = bin_by_slice(slice_idx)
    print("\nWeakest slice-index ranges (lowest share of usable slices, bins with >= 10 slices)")
    for run in runs:
        scored = []
        for b in range(n_bins):
            sel = (bin_id == b) & eligible
            if sel.sum() >= 10:
                scored.append((stats[run]["usable"][sel].to(torch.float64).mean().item(),
                               lo + b * BIN_WIDTH, int(sel.sum())))
        worst = sorted(scored)[:3]
        text = "; ".join(f"slices {start}-{start + BIN_WIDTH - 1}: "
                         f"{share:.0%} usable (n={n})" for share, start, n in worst)
        print(f"  {DISPLAY_NAMES[run]:<22} {text}")

def describe_cases(cases, runs, stats, subject, slice_idx):
    print("\nSelected failure cases")
    for case in cases:
        n = case["index"]
        print(f"\n  Case {case['tag']}: {case['title']}  (subject {subject[n]}, slice {int(slice_idx[n])})")
        print(f"    selected because: {case['reason']}")
        for run in runs:
            s = stats[run]
            dice = "  ".join(f"{SHORT_NAMES[c]} {s['slice_dice'][n, c].item():.3f}" for c in TISSUE
                             if not torch.isnan(s['slice_dice'][n, c]))
            print(f"    {DISPLAY_NAMES[run]:<22} Dice {dice}   boundary-band Dice "
                  f"{s['slice_band_mean'][n].item():.3f}   "
                  f"extra fragments {s['excess'][n].item()}")

#♥=========== Figures (NumPy / Matplotlib) ===========♥#
# Please note: This plotting code was generated by AI.

def to_numpy(tensor):
    return tensor.detach().cpu().numpy()

def label_image(labels):
    return LABEL_COLOURS[labels]

def error_overlay(image, gt, pred):
    """Greyscale MRI with the errors painted on: false positives, false negatives and tissue confusion"""
    rgb = np.repeat(image[..., None] / 255.0, 3, axis=2) * 0.85
    false_positive = (pred > 0) & (gt == 0)
    false_negative = (pred == 0) & (gt > 0)
    confusion = (pred > 0) & (gt > 0) & (pred != gt)
    rgb[false_positive] = list(ERROR_COLOURS.values())[0]
    rgb[false_negative] = list(ERROR_COLOURS.values())[1]
    rgb[confusion] = list(ERROR_COLOURS.values())[2]
    return rgb, int(false_positive.sum()), int(false_negative.sum()), int(confusion.sum())

def clean_axis(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_failure_cases(cases, runs, images, gts, preds, stats, subject, slice_idx, save_path):
    """Rows = failure cases. Columns = input | ground truth | (prediction | errors) for every model"""
    n_cols = 2 + 2 * len(runs)
    fig, axes = plt.subplots(len(cases), n_cols, figsize=(2.9 * n_cols, 3.2 * len(cases) + 0.5), squeeze=False)

    for r, case in enumerate(cases):
        n = case["index"]
        axes[r, 0].imshow(images[n], cmap="gray", vmin=0, vmax=255)
        axes[r, 1].imshow(label_image(gts[n]))
        axes[r, 0].set_ylabel(f"Case {case['tag']}: {case['title']}\nsubject "
                              f"{subject[n]}, slice {slice_idx[n]}", fontsize=9)

        for k, run in enumerate(runs):
            pred_ax, error_ax = axes[r, 2 + 2 * k], axes[r, 3 + 2 * k]
            pred_ax.imshow(label_image(preds[run][n]))
            overlay, n_fp, n_fn, n_conf = error_overlay(images[n], gts[n], preds[run][n])
            error_ax.imshow(overlay)

            dice = " ".join(f"{SHORT_NAMES[c]} {stats[run]['slice_dice'][n, c]:.2f}" for c in TISSUE
                            if not np.isnan(stats[run]["slice_dice"][n, c]))
            pred_ax.set_xlabel(f"Dice: {dice}", fontsize=8)
            error_ax.set_xlabel(f"FP {n_fp} | FN {n_fn} | confused {n_conf} px", fontsize=8)

    titles = ["Input MRI", "Ground truth"]
    for run in runs:
        titles += [f"{DISPLAY_NAMES[run]}\nprediction", f"{DISPLAY_NAMES[run]}\nerrors"]
    for c, title in enumerate(titles):
        axes[0, c].set_title(title, fontsize=10)
    for ax in axes.ravel():
        clean_axis(ax)

    handles = [Patch(facecolor=LABEL_COLOURS[c], edgecolor="grey", label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    handles += [Patch(facecolor=colour, label=name) for name, colour in ERROR_COLOURS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(save_path, dpi=130)
    return fig

def plot_slice_position(runs, stats, slice_idx, eligible, gt_tissue_area, save_path):
    """Where along the head do the models fail? (the brain counterpart of the base / apex slices)"""
    lo, bin_id, n_bins = bin_by_slice(slice_idx)
    centres = lo + (np.arange(n_bins) + 0.5) * BIN_WIDTH

    def per_bin(values):
        out = np.full(n_bins, np.nan)
        for b in range(n_bins):
            sel = (bin_id == b) & eligible & ~np.isnan(values)
            if sel.sum() >= 3:
                out[b] = values[sel].mean()
        return out

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9), gridspec_kw={"height_ratios": [3, 3, 1.6]})
    for run in runs:
        axes[0].plot(centres, per_bin(stats[run]["slice_dice_mean"]), marker="o", ms=3, label=DISPLAY_NAMES[run])
        axes[1].plot(centres, 100 * per_bin(stats[run]["usable"].astype(float)),
                     marker="o", ms=3, label=DISPLAY_NAMES[run])
    axes[0].axhline(GOOD_DICE, color="grey", ls="--", lw=1)
    axes[0].set_ylabel("Mean tissue Dice per slice")
    axes[0].set_title(f"Segmentation quality along the slice axis (bins of {BIN_WIDTH} slices, test set)")
    axes[1].set_ylabel(f"% usable slices\n(all tissues Dice >= {GOOD_DICE})")
    axes[2].fill_between(centres, per_bin(gt_tissue_area.astype(float)), color="grey", alpha=0.5)
    axes[2].set_ylabel("Mean tissue area\n(ground truth, px)")
    axes[2].set_xlabel("Slice index")

    # Shade the "edge" zones
    margin = EDGE_FRACTION * (slice_idx.max() - slice_idx.min())
    for ax in axes:
        ax.axvspan(slice_idx.min(), slice_idx.min() + margin, color="orange", alpha=0.10)
        ax.axvspan(slice_idx.max() - margin, slice_idx.max(), color="orange", alpha=0.10)
        ax.grid(alpha=0.3)
    axes[0].legend(loc="lower center", ncol=len(runs))
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    return fig

def plot_global_vs_boundary(runs, stats, save_path):
    """Global Dice next to boundary-band Dice: how much does a good global score hide?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    width = 0.8 / len(runs)
    x = np.arange(len(TISSUE))
    lowest = 1.0
    for ax, key, title in zip(axes, ["dice", "band_dice"],
                              ["Global Dice (whole slice)",
                               f"Boundary-band Dice (within {BOUNDARY_RADIUS} px of a boundary)"]):
        for k, run in enumerate(runs):
            values = stats[run][key][TISSUE]
            lowest = min(lowest, values.min())
            bars = ax.bar(x + (k - (len(runs) - 1) / 2) * width, values, width, label=DISPLAY_NAMES[run])
            for bar, v in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=7.5)
        ax.axhline(GOOD_DICE, color="grey", ls="--", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([CLASS_NAMES[c] for c in TISSUE])
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylim(max(0.0, lowest - 0.08), 1.02)
    axes[0].set_ylabel("Dice (y-axis does not start at 0)")
    fig.legend(handles=axes[0].get_legend_handles_labels()[0], labels=axes[0].get_legend_handles_labels()[1],
               loc="lower center", ncol=len(runs), frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(save_path, dpi=130)
    return fig

#♥=========== Main ===========♥#
def parse_args():
    parser = argparse.ArgumentParser(description="Run trained models on the test split and analyse their failures")
    parser.add_argument("--models", nargs="+", default=["unet2d", "can_large"], choices=list(RUNS),
                        help="Baseline first, the model under test last")
    parser.add_argument("--weights-dir", type=Path, default=HERE,
                        help="Folder holding best_<run>.pth and results_<run>.json (where train.py ran)")
    parser.add_argument("--out-dir", type=Path, default=HERE / "figures")
    parser.add_argument("--show", action="store_true", help="Also open the figures in windows")
    return parser.parse_args()

def main():
    args = parse_args()
    runs = args.models
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device} | models: {', '.join(runs)}")

    #♥===== Inference =====♥#
    _, _, test_loader = load_data() # Same split as in training: fixed seed, subject-level
    pairs = test_loader.dataset.pairs # The loader does not shuffle, so this is in the same order as the batches
    subject = [p["subject_id"] for p in pairs]
    slice_idx = torch.tensor([p["slice_id"] for p in pairs], dtype=torch.int64)

    images, gts, preds = None, None, {}
    for run in runs:
        model = load_trained_model(run, args.weights_dir, device)
        run_images, run_gts, preds[run] = predict_test_set(model, test_loader, device)
        if images is None:
            images, gts = run_images, run_gts
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(f"Predicted {len(gts)} test slices with {DISPLAY_NAMES[run]}")

    #♥===== Metrics =====♥#
    print("\nAnalysing...")
    band = boundary_band(gts, BOUNDARY_RADIUS)
    gt_components = component_counts(gts, TISSUE, MIN_COMPONENT_PIXELS)
    stats = {run: analyse_model(preds[run], gts, band, gt_components) for run in runs}

    gt_tissue_area = (gts > 0).reshape(len(gts), -1).sum(dim=1)
    eligible = gt_tissue_area >= MIN_SLICE_FG
    lo, hi = int(slice_idx.min()), int(slice_idx.max())
    edge = (slice_idx < lo + EDGE_FRACTION * (hi - lo)) | (slice_idx > hi - EDGE_FRACTION * (hi - lo))

    print(f"\nTest set: {len(gts)} slices from {len(set(subject))} subjects "
          f"({int(eligible.sum())} slices with >= {MIN_SLICE_FG} tissue pixels; slice indices {lo}-{hi})")
    print("Checking against the numbers train.py stored:")
    for run in runs:
        check_against_training(run, stats[run]["dice"], args.weights_dir)

    print_metric_table("Dice (pooled over the test set)", runs, {r: stats[r]["dice"] for r in runs})
    print_metric_table("IoU (pooled over the test set)", runs, {r: stats[r]["iou"] for r in runs})
    print_metric_table(f"Boundary-band Dice (within {BOUNDARY_RADIUS} px of a ground-truth boundary)", runs,
                       {r: stats[r]["band_dice"] for r in runs})
    print_cost_table(runs, args.weights_dir)
    print_usability_summary(runs, stats, slice_idx, edge, eligible)

    #♥===== Failure case autopsy =====♥#
    cases = select_failure_cases(stats, runs, subject, slice_idx, edge, eligible)
    describe_cases(cases, runs, stats, subject, slice_idx)

    # The plotting code works on NumPy arrays, so convert once at the boundary
    images_np, gts_np = to_numpy(images), to_numpy(gts)
    preds_np = {run: to_numpy(p) for run, p in preds.items()}
    stats_np = {run: {key: to_numpy(v) for key, v in s.items()} for run, s in stats.items()}
    slice_idx_np, eligible_np, area_np = to_numpy(slice_idx), to_numpy(eligible), to_numpy(gt_tissue_area)

    figures = [
        plot_failure_cases(cases, runs, images_np, gts_np, preds_np, stats_np, subject,
                           slice_idx_np, args.out_dir / "failure_cases.png"),
        plot_slice_position(runs, stats_np, slice_idx_np, eligible_np, area_np, args.out_dir / "slice_position.png"),
        plot_global_vs_boundary(runs, stats_np, args.out_dir / "global_vs_boundary_dice.png"),
    ]
    print(f"\nSaved figures to {args.out_dir}: failure_cases.png, slice_position.png, global_vs_boundary_dice.png")

    if args.show:
        plt.show()
    for fig in figures:
        plt.close(fig)

if __name__ == "__main__":
    main()