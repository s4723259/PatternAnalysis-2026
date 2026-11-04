# Containing the source code for training, validating, testing and saving the model
# The model should be imported from modules.py and the data loader should be imported
# from dataset.py

# Note: Make sure to plot the losses and metrics during training

#♥=========== Imports ===========♥#
import time
from pathlib import Path

import torch
import matplotlib.pyplot as plt

from dataset import create_image_mask_pairs, split_by_subject, create_dataloaders, IMAGE_SIZE
from utils import confusion_stats, dice_iou_from_stats

#♥=========== Configuration ===========♥#
ROOT = Path(__file__).resolve().parent / "keras_png_slices_data" # Dataset directory

NUM_CLASSES = 4
CLASS_NAMES = ["Background", "CSF", "Gray matter", "White matter"]
BATCH_SIZE = 8
EPOCHS = 30
LEARNING_RATE = 1e-3
RANDOM_SEED = 42

CHECKPOINT_PATH = Path("best_unet2d.pth")
CURVES_PATH = Path("training_curves.png")

#♥=========== Helpers ===========♥#
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_data():
    train_pairs_original = create_image_mask_pairs(ROOT / "keras_png_slices_train",
                                                   ROOT / "keras_png_slices_seg_train")
    val_pairs_original = create_image_mask_pairs(ROOT / "keras_png_slices_validate",
                                                 ROOT / "keras_png_slices_seg_validate")
    test_pairs_original = create_image_mask_pairs(ROOT / "keras_png_slices_test",
                                                  ROOT / "keras_png_slices_seg_test")

    pairs = train_pairs_original + val_pairs_original + test_pairs_original
    print(f"\nTotal pairs MRI slices: {len(pairs)}")

    train_pairs, val_pairs, test_pairs = split_by_subject(
        pairs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=RANDOM_SEED)

    # Multiclass masks are needed for the per-class Dice / IoU
    return create_dataloaders(train_pairs, val_pairs, test_pairs, batch_size=BATCH_SIZE,
                              image_size=IMAGE_SIZE, mask_mode="multiclass")

def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    running_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)

        optimiser.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * images.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns average loss plus per-class Dice and IoU over the whole loader"""
    model.eval()
    running_loss = 0.0
    tp = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    fp = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    fn = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        running_loss += criterion(logits, masks).item() * images.size(0)

        batch_tp, batch_fp, batch_fn = confusion_stats(logits, masks, NUM_CLASSES)
        tp += batch_tp
        fp += batch_fn
        fn += batch_fn

    dice, iou = dice_iou_from_stats(tp, fp, fn)
    return running_loss / len(loader.dataset), dice, iou

@torch.no_grad()
def measure_inference_time(model, loader, device, max_batches=20):
    """Average forward-pass time per slice (in ms), excluding data laoding"""
    model.eval()
    total_time, total_images = 0.0, 0

    for i, (images, _) in enumerate(loader):
        if i >= max_batches + 1:
            break
        images = images.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        if i == 0: # first batch is warm-up
            continue
        total_time += elapsed
        total_images += images.size(0)

    return 1000.0 * total_time / max(total_images, 1)

def plot_history(history, save_path):
    """Plot training/validation loss and validation dice curves"""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Training Loss")
    axes[0].plot(epochs, history["val_loss"], label="Validation Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (CE + Dice)")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["val_mean_dice"], "k--", linewidth=2, label="Mean (foreground)")
    for c in range(NUM_CLASSES):
        per_class = [d[c] for d in history["val_dice_per_class"]]
        axes[1].plot(epochs, per_class, label=CLASS_NAMES[c])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Validation Dice (DSC)")
    axes[1].set_title("Validation Dice")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {save_path}")

