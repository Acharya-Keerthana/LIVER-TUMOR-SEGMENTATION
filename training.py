import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras import backend as K
import pickle
import time

# Set memory growth to prevent TF from allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Define paths
load_dir = '/home/nnm22is069/NEW/livertumor_preprocessed/liver_preprocessed' 
model_save_dir = '/home/nnm22is069/NEW/liver_model'   
os.makedirs(model_save_dir, exist_ok=True)

# Load preprocessed data
print("Loading preprocessed data...")
volumes = np.load(os.path.join(load_dir, 'volumes.npy'))
masks = np.load(os.path.join(load_dir, 'masks.npy'))

try:
    with open(os.path.join(load_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    print(f"Loaded metadata for {len(metadata)} samples")
except:
    print("No metadata file found or error loading it")
    metadata = None

print(f"Data loaded: volumes shape {volumes.shape}, masks shape {masks.shape}")

volumes = volumes.reshape(volumes.shape[0], volumes.shape[1], volumes.shape[2], volumes.shape[3], 1)
print(f"Reshaped volumes: {volumes.shape}")

# Process labels: Convert segmentation masks to binary tumor classification (0: no tumor, 1: tumor)
# Class 2 in segmentation represents tumor
print("Preparing labels...")
tumor_labels = np.array([(mask == 2).any() for mask in masks], dtype=np.float32)
print(f"Tumor distribution: {np.sum(tumor_labels)}/{len(tumor_labels)} samples have tumors")


# Create 3D bounding box coordinates for tumors (for visualization later)
def get_tumor_bounding_boxes(masks):
    bboxes = []
    for mask in masks:
        # Check if tumor exists
        if (mask == 2).any():
            # Get tumor coordinates
            z_indices, y_indices, x_indices = np.where(mask == 2)
            # Create bounding box [z_min, y_min, x_min, z_max, y_max, x_max]
            bbox = [
                np.min(z_indices), np.min(y_indices), np.min(x_indices),
                np.max(z_indices), np.max(y_indices), np.max(x_indices)
            ]
            bboxes.append(bbox)
        else:
            bboxes.append(None)
    return bboxes


# Calculate tumor bounding boxes
tumor_bboxes = get_tumor_bounding_boxes(masks)
print(f"Calculated {sum(1 for b in tumor_bboxes if b is not None)} bounding boxes for tumors")

# Split the data
print("Splitting into train/validation/test sets...")
X_train, X_val_test, y_train, y_val_test = train_test_split(
    volumes, tumor_labels, test_size=0.3, random_state=42, stratify=tumor_labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_val_test, y_val_test, test_size=0.5, random_state=42, stratify=y_val_test
)
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")

# Define Dice coefficient metric for segmentation evaluation
def dice_coefficient(y_true, y_pred):
    """
    Dice Similarity Coefficient metric.
    DSC = 2 * |X n Y| / (|X| + |Y|)
    where X is the predicted set and Y is the ground truth set
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())

# Define IoU (Jaccard index) metric
def iou(y_true, y_pred):
    """
    Intersection over Union (IoU) or Jaccard Index.
    IoU = |X n Y| / |X ? Y| = |X n Y| / (|X| + |Y| - |X n Y|)
    where X is the predicted set and Y is the ground truth set
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + K.epsilon()) / (union + K.epsilon())


# Function to calculate DSC and IoU on numpy arrays (for post-training evaluation)
def calculate_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate DSC and IoU metrics on numpy arrays
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        threshold: Threshold to convert probabilities to binary predictions
    Returns:
        dsc: Dice Similarity Coefficient
        iou_val: Intersection over Union
    """
    y_pred_binary = (y_pred > threshold).astype(np.float32)

    # Flatten arrays
    y_true = y_true.flatten()
    y_pred_binary = y_pred_binary.flatten()

    # Calculate intersection and union
    intersection = np.sum(y_true * y_pred_binary)
    sum_y_true = np.sum(y_true)
    sum_y_pred = np.sum(y_pred_binary)

    # Calculate DSC
    dsc = (2. * intersection + 1e-7) / (sum_y_true + sum_y_pred + 1e-7)

    # Calculate IoU
    union = sum_y_true + sum_y_pred - intersection
    iou_val = (intersection + 1e-7) / (union + 1e-7)

    return dsc, iou_val

def create_lightweight_3d_cnn(input_shape):
    """
    Create a lightweight 3D CNN model for liver tumor classification.
    """
    inputs = layers.Input(input_shape)

    # First 3D convolution block - reduce spatial dimensions
    x = layers.Conv3D(16, kernel_size=3, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Second 3D convolution block
    x = layers.Conv3D(32, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Third 3D convolution block
    x = layers.Conv3D(64, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling3D(pool_size=(2, 2, 2))(x)

    # Global average pooling to reduce parameters
    x = layers.GlobalAveragePooling3D()(x)

    # Dense layers
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer for binary classification (tumor/no tumor)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)
    return model

# Create and compile the model
input_shape = X_train.shape[1:]  # (depth, height, width)
print(f"Input shape: {input_shape}")

model = create_lightweight_3d_cnn(input_shape)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.AUC(),
        dice_coefficient,  # Add DSC metric
        iou               # Add IoU metric
    ]
)

model.summary()

# Add this code here, after model.summary() but before defining callbacks

import os
from tensorflow.keras.models import load_model

# Path to your checkpoint
checkpoint_path = os.path.join(model_save_dir, 'liver_tumor_model_best.keras')

# Check if a checkpoint exists
if os.path.exists(checkpoint_path):
    print(f"Loading existing model from {checkpoint_path}")
    # Load the model with custom metrics
    custom_objects = {
        'dice_coefficient': dice_coefficient,
        'iou': iou
    }
    model = load_model(checkpoint_path, custom_objects=custom_objects)
    
    # Set the initial epoch to continue from
    initial_epoch = 12  # Since 12 epochs completed, start from 13th
else:
    print("No checkpoint found. Training from scratch.")
    initial_epoch = 0

# Callbacks
callbacks = [
    ModelCheckpoint(
        filepath=os.path.join(model_save_dir, 'liver_tumor_model_best.keras'),
        monitor='val_dice_coefficient',  # Monitor DSC instead of AUC
        mode='max',
        save_best_only=True,
        verbose=1
    ),
    EarlyStopping(
        monitor='val_dice_coefficient',  # Monitor DSC instead of AUC
        patience=10,
        restore_best_weights=True,
        verbose=1,
	mode='max'
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1,
	mode='min'
    )
]

# Train the model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    initial_epoch=initial_epoch,  
    batch_size=4,  
    callbacks=callbacks,
    verbose=1
)

# Save the final model
model.save(os.path.join(model_save_dir, 'liver_tumor_model_final.keras'))
print(f"Model saved to {os.path.join(model_save_dir, 'liver_tumor_model_final.keras')}")

# Function to calculate tumor severity based on volume and size
def calculate_tumor_severity(mask):
    """
    Calculate tumor severity score based on:
    1. Tumor volume relative to liver volume
    2. Maximum tumor dimensions

    Returns a severity score from 0-10 and a category
    """
    if not (mask == 2).any():
        return 0, "No tumor detected"

    # Calculate volumes
    liver_volume = np.sum(mask == 1)
    tumor_volume = np.sum(mask == 2)

    # Calculate tumor percentage of liver
    if liver_volume > 0:
        tumor_percent = (tumor_volume / liver_volume) * 100
    else:
        tumor_percent = 0

    # Get tumor dimensions
    z_indices, y_indices, x_indices = np.where(mask == 2)
    z_size = np.max(z_indices) - np.min(z_indices)
    y_size = np.max(y_indices) - np.min(y_indices)
    x_size = np.max(x_indices) - np.min(x_indices)
    max_dimension = max(z_size, y_size, x_size)

    # Calculate severity score (0-10)
    # This is a simplified scoring approach - adjust based on medical criteria
    volume_score = min(10, tumor_percent * 0.5)
    size_score = min(10, max_dimension * 0.2)

    # Combined score - weighted average
    severity_score = (volume_score * 0.7) + (size_score * 0.3)

    # Determine category
    if severity_score < 2:
        category = "Very mild"
    elif severity_score < 4:
        category = "Mild"
    elif severity_score < 6:
        category = "Moderate"
    elif severity_score < 8:
        category = "Severe"
    else:
        category = "Very severe"

    return round(severity_score, 1), category

# Process test set samples to create sample visualization data
print("Processing test samples for visualization...")
test_results = []

# Calculate DSC and IoU for each test sample
dsc_scores = []
iou_scores = []

for i in range(min(10, len(X_test))):  # Process up to 10 test samples
    # Get predictions
    pred = model.predict(np.expand_dims(X_test[i], axis=0))[0][0]

    # Calculate DSC and IoU for this sample
    sample_dsc, sample_iou = calculate_metrics(np.array([y_test[i]]), np.array([pred]))
    dsc_scores.append(sample_dsc)
    iou_scores.append(sample_iou)

    sample_data = {
        'volume': X_test[i],
        'actual_label': int(y_test[i]),
        'predicted_prob': float(pred),
        'predicted_label': 1 if pred > 0.5 else 0,
        'dsc': float(sample_dsc),
        'iou': float(sample_iou)
    }

    # Get corresponding index in original dataset
    original_idx = len(X_train) + len(X_val) + i

    # Add bounding box if tumor exists
    if tumor_bboxes[original_idx] is not None:
        sample_data['tumor_bbox'] = tumor_bboxes[original_idx]

        # Calculate severity if we have the masks
        mask_idx = len(X_train) + len(X_val) + i
        if mask_idx < len(masks):
            severity_score, severity_category = calculate_tumor_severity(masks[mask_idx])
            sample_data['severity_score'] = severity_score
            sample_data['severity_category'] = severity_category

    test_results.append(sample_data)

# Calculate and print average DSC and IoU for the test set
print(f"Average DSC on test set: {np.mean(dsc_scores):.4f}")
print(f"Average IoU on test set: {np.mean(iou_scores):.4f}")

# Save visualization data for the Streamlit app
with open(os.path.join(model_save_dir, 'test_visualization_data.pkl'), 'wb') as f:
    pickle.dump(test_results, f)

# Evaluate model
print("Evaluating model on test set...")
test_metrics = model.evaluate(X_test, y_test)
test_loss, test_acc, test_auc, test_dsc, test_iou = test_metrics
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test DSC: {test_dsc:.4f}")
print(f"Test IoU: {test_iou:.4f}")

# Plot training history
plt.figure(figsize=(15, 10))

# Plot Loss
plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

# Plot Accuracy
plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot DSC
plt.subplot(2, 2, 3)
plt.plot(history.history['dice_coefficient'], label='Training DSC')
plt.plot(history.history['val_dice_coefficient'], label='Validation DSC')
plt.xlabel('Epoch')
plt.ylabel('Dice Coefficient')
plt.legend()
plt.title('Training and Validation Dice Coefficient')

# Plot IoU
plt.subplot(2, 2, 4)
plt.plot(history.history['iou'], label='Training IoU')
plt.plot(history.history['val_iou'], label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.legend()
plt.title('Training and Validation IoU')

plt.tight_layout()
plt.savefig(os.path.join(model_save_dir, 'training_history.png'))
plt.close()

# Create detailed performance visualization
plt.figure(figsize=(10, 8))
thresholds = np.arange(0.1, 1.0, 0.1)
dsc_values = []
iou_values = []

# Calculate DSC and IoU at different thresholds
for threshold in thresholds:
    y_pred = model.predict(X_test)
    threshold_dsc, threshold_iou = calculate_metrics(y_test, y_pred, threshold)
    dsc_values.append(threshold_dsc)
    iou_values.append(threshold_iou)

# Plot metrics vs thresholds
plt.plot(thresholds, dsc_values, 'b-', label='DSC')
plt.plot(thresholds, iou_values, 'r-', label='IoU')
plt.xlabel('Classification Threshold')
plt.ylabel('Metric Value')
plt.title('DSC and IoU at Different Classification Thresholds')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_save_dir, 'dsc_iou_thresholds.png'))
plt.close()

print("Model training complete!")

