# Contains the data loader for loading and preprocessing data, including leakage free
# training/val/test splitting functions and data augmentations

#♥=========== Imports ===========♥#
import os
import re
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

#♥=========== Data Configuration ===========♥#
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
RANDOM_SEED = 42
MASK_MODE = "binary" # This is for changing the segmentation task between binary and multiple class


#♥=========== Pair images with the segmentation masks ===========♥#
FILENAME_PATTERN = re.compile(r"^(case|seg)_(\d+)_slice_(\d+)\.nii\.png$")

def extract_ids(filename):
    """
    This extracts the subject ID and slice number from the filename
    :param filename: eg.) case_402_slice_4.nii.png
    :return:
        subject_id = 402
        slice_id = 4
    """

    match = FILENAME_PATTERN.match(filename)
    if match is None:
        return None

    subject_id = int(match.group(2))
    slice_id = int(match.group(3))

    return subject_id, slice_id

def create_image_mask_pairs(image_dir, mask_dir):
    """
    This matches the MRI images with the corresponding segmentation masks
    :param image_dir:
    :param mask_dir:
    :return:
        A list of dictionaries containing:
            image path
            mask path
            subject ID
            slice ID
    """

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not image_dir.exists() or not mask_dir.exists(): # Incase files aren't there
        raise FileNotFoundError(
            f"Image or mask directory does not exist: {image_dir} and {mask_dir}"
        )

    # Collect image files
    image_files = {}
    for filename in os.listdir(image_dir):
        ids = extract_ids(filename)

        if ids is None: continue
        if not filename.startswith("case_"): continue
        if ids in image_files: raise ValueError(f"Duplicate MRI image: {filename}")

        image_files[ids] = image_dir / filename

    # Collect segmentation masks
    mask_files = {}
    for filename in os.listdir(mask_dir):
        ids = extract_ids(filename)

        if ids is None: continue
        if not filename.startswith("seg_"): continue
        if ids in mask_files: raise ValueError(f"Duplicate segmentation mask: {filename}")

        mask_files[ids] = mask_dir / filename

    # Double check that image and mask sets match
    image_ids = set(image_files.keys())
    mask_ids = set(mask_files.keys())
    missing_masks = image_ids - mask_ids
    missing_images = mask_ids - image_ids

    if missing_masks:
        raise ValueError(f"Missing masks for {len(missing_masks)} MRI slices: "
                         f"{list(missing_masks)[:5]}")
    if missing_images:
        raise ValueError(f"Found {len(missing_images)} masks without MRI images: "
                         f"{list(missing_images)[:5]}"
                         )

    # Create matched pairs
    pairs = []
    for key in sorted(image_ids):
        subject_id, slice_id = key
        pairs.append({
            "image": image_files[key],
            "mask": mask_files[key],
            "subject_id": subject_id,
            "slice_id": slice_id
        })
    if not pairs:
        raise ValueError(f"No valid image-mask pairs found")

    print(f"Successfully paired {len(pairs)} MRI slices")
    return pairs

#♥=========== Subject-level data splitting ===========♥#
# Note: The provided data is already split by subject, however we implement splitting regardless to standardise all
#       inputted data
def split_by_subject(pairs, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    This splits the dataset by subject rather than individual slices~ this is done as
    slices belonging to the same patient must not appear in multiple dataset partitions
    """

    assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0)
    subjects = sorted({pair["subject_id"] for pair in pairs})
    rng = np.random.default_rng(seed)

    subjects = np.array(subjects)
    rng.shuffle(subjects)

    n_subjects = len(subjects)
    n_train = int(n_subjects * train_ratio)
    n_val = int(n_subjects * val_ratio)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train+n_val])
    test_subjects = set(subjects[n_train+n_val:])

    train_pairs = []
    val_pairs = []
    test_pairs = []

    for pair in pairs:
        subject = pair["subject_id"]
        if subject in train_subjects:
            train_pairs.append(pair)
        elif subject in val_subjects:
            val_pairs.append(pair)
        else:
            test_pairs.append(pair)

    # Verify there's no subject overlap
    assert train_subjects.isdisjoint(val_subjects)
    assert train_subjects.isdisjoint(test_subjects)
    assert val_subjects.isdisjoint(test_subjects)

    print("Dataset splitting complete!")
    print("♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥♥")
    print(f"Total subjects: {n_subjects}")
    print(f"Training subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")

    print(f"Training slices: {len(train_pairs)}")
    print(f"Validation Slices: {len(val_pairs)}")
    print(f"Test Slices: {len(test_pairs)}")

    return train_pairs, val_pairs, test_pairs


#♥=========== Image Preprocessing + Data Augmentation ===========♥#

def preprocess_image(image, mask, image_size=(256, 256)):
    """
    This resizes the MRI image and segmentation mask~

    Note that the MRI images use bilinear interpolation.
    Segmentation masks use nearest neighbour interpolation to preserve
    their discrete class labels
    """

    image = image.resize(image_size, Image.Resampling.BILINEAR)
    mask = mask.resize(image_size, Image.Resampling.NEAREST)

    return image, mask

def augment_image(image, mask):
    """
    Applies identical spatial transformations to image and mask

    Also future proof for intensity transformations, applied to MRI images
    """

    # Random horizontal flip
    if random.random() < 0.5:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

    # Small random rotation
    angle = random.uniform(-7, 7)
    image = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
    mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)

    return image, mask


#♥=========== PyTorch Dataset Class ===========♥#

class OASISDataset(Dataset):
    def __init__(
            self,
            pairs,
            image_size = (256, 256),
            augment = False,
            mask_mode = "binary"
    ):
        self.pairs = pairs
        self.image_size = image_size
        self.augment = augment
        self.mask_mode = mask_mode

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        # Load the MRI image and corresponding segmentation
        image = Image.open(pair["image"]).convert("L")
        mask = Image.open(pair["mask"])

        # Resize the image and mask
        image, mask = preprocess_image(image, mask, self.image_size)

        # Augment the training data
        if self.augment:
            image, mask = augment_image(image, mask)

        # Convert images to NumPy arrays
        image = np.array(image, dtype = np.float32) / 255.0
        mask = np.array(mask)

        # Convert image into PyTorch tensor
        image = torch.from_numpy(image).unsqueeze(0)

        # Convert segmentation mask
        if self.mask_mode == "binary":
            # Assume background = 0 and foreground > 0
            mask = (mask > 0).astype(np.float32)
            mask = torch.from_numpy(mask).unsqueeze(0)
        elif self.mask_mode == "multiclass":
            mask = torch.from_numpy(mask.astype(np.int64))

        else:
            raise ValueError(f"mask_mode must be 'binary' or 'multiclass'")

        return image, mask


#♥=========== Create the pytorch dataloaders ===========♥#

def create_dataloaders(train_pairs, val_pairs, test_pairs, batch_size=8,
                       image_size=(256, 256), mask_mode="binary"):
    train_dataset = OASISDataset(train_pairs, image_size, augment=True, mask_mode=mask_mode)
    val_dataset = OASISDataset(val_pairs, image_size, augment=False, mask_mode=mask_mode)
    test_dataset = OASISDataset(test_pairs, image_size, augment=False, mask_mode=mask_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader