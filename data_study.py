import os
import zipfile
import numpy as np
import nibabel as nib
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.keras.backend as K
from tqdm import tqdm
import glob
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

zip_path = '/home/nnm22is069/NEW/archive.zip'
extract_path = '/home/nnm22is069/NEW/liver_dataset'

# Only extract if the folder doesn't already exist
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction done.")
else:
    print("Already extracted.")

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load and transpose to get correct axis orientation
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Load and transpose to get correct axis orientation
volume_path = '/home/nnm22is069/NEW/liver_dataset/volume_pt1/volume-0.nii'
seg_path = '/home/nnm22is069/NEW/liver_dataset/segmentations/segmentation-0.nii'

volume = nib.load(volume_path).get_fdata().transpose(2,1,0)
segmentation = nib.load(seg_path).get_fdata().transpose(2,1,0)

# Find slice indices with non-zero segmentation
non_zero_slices = np.unique(np.where(segmentation != 0)[0])
print(f"Found {len(non_zero_slices)} slices with liver/tumor segmentation.")

# Pick a middle slice with liver
slice_idx = non_zero_slices[len(non_zero_slices)//2]

# Plot
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(volume[slice_idx], cmap='gray')
axes[0].set_title('CT Scan Slice')
axes[0].axis('off')

axes[1].imshow(volume[slice_idx], cmap='gray')
axes[1].imshow(segmentation[slice_idx], cmap='Reds', alpha=0.5)
axes[1].set_title('Segmentation Overlay')
axes[1].axis('off')

plt.tight_layout()
output_path = '/home/nnm22is069/NEW/slice_visualization.png'
plt.savefig(output_path)
print(f"Plot saved to {output_path}")


print("Volume shape:", volume.shape)
print("Segmentation shape:", segmentation.shape)
print("Volume intensity range:", volume.min(), "to", volume.max())
print("Segmentation unique values:", set(segmentation.flatten()))

def normalize(volume):
    volume = np.clip(volume, -100, 400)
    volume = (volume - (-100)) / (400 - (-100))
    return volume


import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

# Paths
volume_dir = '/home/nnm22is069/NEW/liver_dataset/volume_pt1'
seg_dir = '/home/nnm22is069/NEW/liver_dataset/segmentations'

# Limit to the first 10 matching volume-segmentation pairs
num_samples = 10

# Create a figure
fig, axes = plt.subplots(num_samples, 2, figsize=(10, num_samples * 2.5))
plt.subplots_adjust(top=0.97)

for i in range(num_samples):
    vol_path = os.path.join(volume_dir, f'volume-{i}.nii')
    seg_path = os.path.join(seg_dir, f'segmentation-{i}.nii')

    if not os.path.exists(vol_path) or not os.path.exists(seg_path):
        print(f"Missing volume or segmentation for index {i}. Skipping...")
        continue

    # Load volume and mask, transpose to get (slices, H, W)
    volume = nib.load(vol_path).get_fdata().transpose(2, 1, 0)
    segmentation = nib.load(seg_path).get_fdata().transpose(2, 1, 0)

    # Find a slice with liver/tumor
    non_zero_slices = np.unique(np.where(segmentation != 0)[0])
    if len(non_zero_slices) == 0:
        print(f"No segmentation in sample {i}. Skipping...")
        continue

    slice_idx = non_zero_slices[len(non_zero_slices) // 2]

    # Plot CT slice
    axes[i, 0].imshow(volume[slice_idx], cmap='gray')
    axes[i, 0].set_title(f'CT Slice #{i}')
    axes[i, 0].axis('off')

    # Plot segmentation overlay
    axes[i, 1].imshow(volume[slice_idx], cmap='gray')
    liver = segmentation[slice_idx] == 1
    tumor = segmentation[slice_idx] == 2
    axes[i, 1].imshow(liver, cmap='Reds', alpha=0.4)
    axes[i, 1].imshow(tumor, cmap='Blues', alpha=0.6)
    axes[i, 1].set_title(f'Segmentation #{i}')
    axes[i, 1].axis('off')

    plt.tight_layout()
    output_path = '/home/nnm22is069/NEW/images.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

# Base dataset path
volume_base = '/home/nnm22is069/NEW/liver_dataset'
segmentation_dir = os.path.join(volume_base, 'segmentations')

# Volume folders with file index mapping - extended to include all parts
volume_dirs = {
    'volume_pt1': range(0, 11),  # Include 0-10
    'volume_pt2': range(10, 21), # Include 10-20
    'volume_pt3': range(20, 31), # Include 20-30
    'volume_pt4': range(30, 41), # Include 30-40
    'volume_pt5': range(40, 51)  # Include 40-50
}

# Preprocessing Parameters
TARGET_SPACING = [1.5, 1.5, 1.5]
HU_MIN, HU_MAX = -100, 400
CROP_MARGIN = 10
APPLY_NOISE_REDUCTION = True
APPLY_Z_SCORE = False

def normalize(volume):
    """Normalize volume according to HU windowing and min-max normalization"""
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume

def find_volume_path(idx):
    """
    Helper function to find the correct volume path for a given index
    by checking all volume folders
    """
    for folder, indices in volume_dirs.items():
        if idx in indices:
            vol_path = os.path.join(volume_base, folder, f'volume-{idx}.nii')
            if os.path.exists(vol_path):
                return vol_path
    return None

def find_available_pairs():
    """
    Find all available pairs of volumes and segmentations
    Returns a list of (volume_path, segmentation_path, index) tuples
    """
    pairs = []

    # Find all segmentation files
    seg_files = [f for f in os.listdir(segmentation_dir) if f.startswith('segmentation-') and f.endswith('.nii')]

    # For each segmentation, check if there's a matching volume
    for seg_file in seg_files:
        try:
            idx = int(seg_file.split('-')[1].split('.')[0])
            vol_path = find_volume_path(idx)

            if vol_path is not None:
                seg_path = os.path.join(segmentation_dir, seg_file)
                pairs.append((vol_path, seg_path, idx))
        except Exception as e:
            print(f"Error processing {seg_file}: {e}")
            continue

    # Sort by index
    pairs.sort(key=lambda x: x[2])

    return pairs

def preprocess_liver_ct(volume_path, segmentation_path):
    """
    Comprehensive preprocessing pipeline for liver CT volumes and segmentations

    Parameters:
    - volume_path: Path to the CT scan (.nii or .nii.gz)
    - segmentation_path: Path to the segmentation mask (.nii or .nii.gz)

    Returns:
    - vol_preprocessed: Preprocessed CT volume (numpy array)
    - seg_preprocessed: Preprocessed segmentation mask (numpy array)
    - metadata: Dictionary with preprocessing metadata
    """
    try:
        # Load the volume and segmentation using SimpleITK
        vol_sitk = sitk.ReadImage(volume_path)
        seg_sitk = sitk.ReadImage(segmentation_path)

        # Get original spacing
        original_spacing = vol_sitk.GetSpacing()

        # Step 1: Resample to isotropic voxel spacing
        vol_resampled, seg_resampled = resample_volume_and_mask(vol_sitk, seg_sitk, TARGET_SPACING)

        # Convert to numpy arrays for further processing
        vol_np = sitk.GetArrayFromImage(vol_resampled).transpose(2, 1, 0)  # Convert to [D, H, W]
        seg_np = sitk.GetArrayFromImage(seg_resampled).transpose(2, 1, 0).astype(np.uint8)

        # Step 2: Optional noise reduction
        if APPLY_NOISE_REDUCTION:
            vol_np = gaussian_filter(vol_np, sigma=0.5)

        # Step 3: HU windowing + Min-Max normalization
        vol_np = normalize(vol_np)

        # Step 4: ROI cropping based on liver/tumor segmentation
        vol_crop, seg_crop = crop_to_roi(vol_np, seg_np, margin=CROP_MARGIN)

        # Step 5: Optional Z-score standardization
        if APPLY_Z_SCORE:
            # Use non-zero voxels (or liver voxels if available) for mean/std calculation
            if np.any(seg_crop == 1):  # If liver segmentation exists
                mask = seg_crop == 1
            else:
                mask = vol_crop > 0

            mean = vol_crop[mask].mean()
            std = vol_crop[mask].std()

            # Apply z-score normalization
            vol_crop = (vol_crop - mean) / (std + 1e-8)  # Add epsilon to avoid division by zero

        # Collect metadata
        metadata = {
            'original_shape': sitk.GetArrayFromImage(vol_sitk).shape,
            'original_spacing': original_spacing,
            'resampled_shape': vol_np.shape,
            'resampled_spacing': TARGET_SPACING,
            'cropped_shape': vol_crop.shape,
            'hu_range': (HU_MIN, HU_MAX)
        }

        return vol_crop.astype(np.float32), seg_crop, metadata

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return None, None, None
def resample_volume_and_mask(vol_sitk, seg_sitk, target_spacing):
    """
    Resample volume and segmentation to target spacing

    Parameters:
    - vol_sitk: SimpleITK image of the CT volume
    - seg_sitk: SimpleITK image of the segmentation mask
    - target_spacing: Target voxel spacing [x, y, z] in mm

    Returns:
    - vol_resampled: Resampled volume
    - seg_resampled: Resampled segmentation
    """
    # Get original size and spacing
    original_spacing = vol_sitk.GetSpacing()
    original_size = vol_sitk.GetSize()

    # Calculate new size
    new_size = [
        int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
        int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
        int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
    ]

    # Resample volume using linear interpolation
    resample_vol = sitk.ResampleImageFilter()
    resample_vol.SetOutputSpacing(target_spacing)
    resample_vol.SetSize(new_size)
    resample_vol.SetOutputDirection(vol_sitk.GetDirection())
    resample_vol.SetOutputOrigin(vol_sitk.GetOrigin())
    resample_vol.SetTransform(sitk.Transform())
    resample_vol.SetDefaultPixelValue(vol_sitk.GetPixelIDValue())
    resample_vol.SetInterpolator(sitk.sitkLinear)
    vol_resampled = resample_vol.Execute(vol_sitk)

    # Resample segmentation using nearest neighbor interpolation
    resample_seg = sitk.ResampleImageFilter()
    resample_seg.SetOutputSpacing(target_spacing)
    resample_seg.SetSize(new_size)
    resample_seg.SetOutputDirection(seg_sitk.GetDirection())
    resample_seg.SetOutputOrigin(seg_sitk.GetOrigin())
    resample_seg.SetTransform(sitk.Transform())
    resample_seg.SetDefaultPixelValue(0)
    resample_seg.SetInterpolator(sitk.sitkNearestNeighbor)
    seg_resampled = resample_seg.Execute(seg_sitk)

    return vol_resampled, seg_resampled

def crop_to_roi(volume, segmentation, margin=10):
    """
    Crop volume and segmentation to the ROI defined by non-zero segmentation values

    Parameters:
    - volume: CT volume as numpy array [D, H, W]
    - segmentation: Segmentation mask as numpy array [D, H, W]
    - margin: Extra margin around the ROI (in voxels)

    Returns:
    - vol_crop: Cropped volume
    - seg_crop: Cropped segmentation
    """
    # Find non-zero segmentation indices
    if np.any(segmentation > 0):
        z_indices, y_indices, x_indices = np.where(segmentation > 0)

        # Get bounding box with margin
        z_min, z_max = max(0, z_indices.min() - margin), min(volume.shape[0], z_indices.max() + margin + 1)
        y_min, y_max = max(0, y_indices.min() - margin), min(volume.shape[1], y_indices.max() + margin + 1)
        x_min, x_max = max(0, x_indices.min() - margin), min(volume.shape[2], x_indices.max() + margin + 1)

        # Crop volume and segmentation
        vol_crop = volume[z_min:z_max, y_min:y_max, x_min:x_max]
        seg_crop = segmentation[z_min:z_max, y_min:y_max, x_min:x_max]

        return vol_crop, seg_crop
    else:
        # If no segmentation found, return the original volume
        print("No segmentation found. Returning uncropped volume.")
        return volume, segmentation

def pad_volume_to_size(volume, target_shape):
    """
    Pad volume to target shape

    Parameters:
    - volume: Volume to pad
    - target_shape: Target shape (D, H, W)

    Returns:
    - padded_volume: Padded volume
    """
    # Calculate padding for each dimension
    pad_d = max(0, target_shape[0] - volume.shape[0])
    pad_h = max(0, target_shape[1] - volume.shape[1])
    pad_w = max(0, target_shape[2] - volume.shape[2])

    # Calculate padding before and after for each dimension
    pad_d_before = pad_d // 2
    pad_d_after = pad_d - pad_d_before

    pad_h_before = pad_h // 2
    pad_h_after = pad_h - pad_h_before

    pad_w_before = pad_w // 2
    pad_w_after = pad_w - pad_w_before

    # Pad the volume
    padding = ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after))
    padded_volume = np.pad(volume, padding, mode='constant', constant_values=0)

    return padded_volume


def visualize_preprocessed_sample(original_vol, original_seg, processed_vol, processed_seg):
    """
    Visualize the original and preprocessed data
    """
    # Find middle slices with segmentation
    orig_non_zero = np.unique(np.where(original_seg > 0)[0])
    proc_non_zero = np.unique(np.where(processed_seg > 0)[0])

    if len(orig_non_zero) == 0 or len(proc_non_zero) == 0:
        print("No segmentation found in either original or processed volume.")
        orig_slice = original_vol.shape[0] // 2
        proc_slice = processed_vol.shape[0] // 2
    else:
        orig_slice = orig_non_zero[len(orig_non_zero) // 2]
        proc_slice = proc_non_zero[len(proc_non_zero) // 2]

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original volume
    axes[0, 0].imshow(original_vol[orig_slice], cmap='gray')
    axes[0, 0].set_title('Original CT')
    axes[0, 0].axis('off')

    # Original segmentation overlay
    axes[0, 1].imshow(original_vol[orig_slice], cmap='gray')
    liver_mask = original_seg[orig_slice] == 1
    tumor_mask = original_seg[orig_slice] == 2
    axes[0, 1].imshow(liver_mask, cmap='Reds', alpha=0.4)
    axes[0, 1].imshow(tumor_mask, cmap='Blues', alpha=0.6)
    axes[0, 1].set_title('Original Segmentation')
    axes[0, 1].axis('off')

    # Processed volume
    axes[1, 0].imshow(processed_vol[proc_slice], cmap='gray')
    axes[1, 0].set_title('Preprocessed CT')
    axes[1, 0].axis('off')

    # Processed segmentation overlay
    axes[1, 1].imshow(processed_vol[proc_slice], cmap='gray')
    liver_mask = processed_seg[proc_slice] == 1
    tumor_mask = processed_seg[proc_slice] == 2
    axes[1, 1].imshow(liver_mask, cmap='Reds', alpha=0.4)
    axes[1, 1].imshow(tumor_mask, cmap='Blues', alpha=0.6)
    axes[1, 1].set_title('Preprocessed Segmentation')
    axes[1, 1].axis('off')

    plt.tight_layout()
    output_path = '/home/nnm22is069/NEW/preprocessed_image.png'
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

def process_all_samples():
    """
    Process all available samples in the dataset
    """
    volumes_preprocessed = []
    masks_preprocessed = []
    metadata_list = []
    original_shapes = []

    # Find all matching volume-segmentation pairs
    pairs = find_available_pairs()

    print(f"Found {len(pairs)} matching volume-segmentation pairs.")

    # Process each pair
    for vol_path, seg_path, idx in tqdm(pairs, desc="Processing samples"):
        try:
            # Load original data for visualization
            orig_vol = nib.load(vol_path).get_fdata().transpose(2, 1, 0)
            orig_seg = nib.load(seg_path).get_fdata().transpose(2, 1, 0)

            # Apply preprocessing pipeline
            vol_processed, seg_processed, metadata = preprocess_liver_ct(vol_path, seg_path)

            if vol_processed is None:
                continue

            # Store shape for calculating max dimensions
            original_shapes.append(vol_processed.shape)

            # Save preprocessed data
            volumes_preprocessed.append(vol_processed)
            masks_preprocessed.append(seg_processed)
            metadata_list.append(metadata)

            # Show progress every 10 samples
            if idx % 10 == 0:
                print(f"\nSample {idx}:")
                print(f"  Original shape: {orig_vol.shape}, Preprocessed shape: {vol_processed.shape}")
                visualize_preprocessed_sample(orig_vol, orig_seg, vol_processed, seg_processed)

        except Exception as e:
            print(f"Error processing sample {idx}: {e}")

    # Calculate the maximum dimensions needed
    if original_shapes:
        max_d = max([shape[0] for shape in original_shapes])
        max_h = max([shape[1] for shape in original_shapes])
        max_w = max([shape[2] for shape in original_shapes])

        # Round up to nearest multiple of 16 (common for deep learning)
        max_d = ((max_d + 15) // 16) * 16
        max_h = ((max_h + 15) // 16) * 16
        max_w = ((max_w + 15) // 16) * 16

        target_shape = (max_d, max_h, max_w)
        print(f"\nPadding all volumes to shape: {target_shape}")

        # Pad all volumes and masks to the same size
        padded_volumes = []
        padded_masks = []

        for vol, mask in zip(volumes_preprocessed, masks_preprocessed):
            padded_vol = pad_volume_to_size(vol, target_shape)
            padded_mask = pad_volume_to_size(mask, target_shape)
            padded_volumes.append(padded_vol)
            padded_masks.append(padded_mask)

        return padded_volumes, padded_masks, metadata_list, target_shape

    return volumes_preprocessed, masks_preprocessed, metadata_list, None

# Run the preprocessing pipeline
if __name__ == "__main__":
    # Show the paths we're using
    print(f"Volume base path: {volume_base}")
    print(f"Segmentation directory: {segmentation_dir}")
    print("Volume directories:")
    for folder, indices in volume_dirs.items():
        folder_path = os.path.join(volume_base, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.startswith('volume-') and f.endswith('.nii')]
            print(f"  {folder}: {len(files)} files, indices {min(indices)}-{max(indices)}")
        else:
            print(f"  {folder}: Directory not found")

    # Count total volume files
    total_volumes = 0
    for folder in volume_dirs.keys():
        folder_path = os.path.join(volume_base, folder)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.startswith('volume-') and f.endswith('.nii')]
            total_volumes += len(files)

    # Check how many segmentation files we have
    seg_files = [f for f in os.listdir(segmentation_dir) if f.startswith('segmentation-') and f.endswith('.nii')]
    print(f"Total volume files: {total_volumes}")
    print(f"Total segmentation files: {len(seg_files)}")

    # Process all samples
    volumes, masks, metadata, target_shape = process_all_samples()

    print(f"\nPreprocessed {len(volumes)} volumes successfully")
    print(f"All volumes padded to shape: {target_shape}")

    if volumes:
        # Calculate statistics
        mean_intensity = np.mean([np.mean(vol) for vol in volumes])
        std_intensity = np.mean([np.std(vol) for vol in volumes])

        print(f"Mean volume intensity: {mean_intensity:.4f}")
        print(f"Mean volume standard deviation: {std_intensity:.4f}")


