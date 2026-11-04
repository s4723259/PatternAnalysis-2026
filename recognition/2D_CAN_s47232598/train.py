# Containing the source code for training, validating, testing and saving the model
# The model should be imported from modules.py and the data loader should be imported
# from dataset.py

# Note: Make sure to plot the losses and metrics during training


#♥=========== Dataset Importing ===========♥#
from dataset import (create_image_mask_pairs, split_by_subject, create_dataloaders)
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "keras_png_slices_data" # Dataset directory

train_image_dir = ROOT / "keras_png_slices_train"
train_mask_dir = ROOT / "keras_png_slices_seg_train"

val_image_dir = ROOT / "keras_png_slices_validate"
val_mask_dir = ROOT / "keras_png_slices_seg_validate"

test_image_dir = ROOT / "keras_png_slices_test"
test_mask_dir = ROOT / "keras_png_slices_seg_test"

# Pair the MRI images with segmentation masks
train_pairs_original = create_image_mask_pairs(train_image_dir, train_mask_dir)
val_pairs_original = create_image_mask_pairs(val_image_dir, val_mask_dir)
test_pairs_original = create_image_mask_pairs(test_image_dir, test_mask_dir)

# Combine all the pairs before resplitting
pairs = (train_pairs_original + val_pairs_original + test_pairs_original)
print(f"\nTotal pairs MRI slices: {len(pairs)}")

# Split by subject, not by slice
train_pairs, val_pairs, test_pairs = split_by_subject(pairs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42)

# Create dataloaders
train_loader, val_loader, test_loader = create_dataloaders(train_pairs, val_pairs, test_pairs, batch_size=8,
                                                           image_size=(256, 256), mask_mode="binary")

# Check output dimensions
images, masks = next(iter(train_loader))

print("\nDataLoader Verification")
print("♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥")
print("Image batch shape:", images.shape)
print("Mask batch shape:", masks.shape)