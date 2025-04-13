import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.patches import Rectangle
import os
import pickle
import nibabel as nib
import io
from PIL import Image
import SimpleITK as sitk
from scipy.ndimage import gaussian_filter
import tempfile
import shutil
import time


# Set page configuration
st.set_page_config(
    page_title="Liver Tumor Detection",
    page_icon="??",
    layout="wide"
)

# Global variables
MODEL_PATH = 'liver_model/liver_tumor_model_final.keras'
VIZ_DATA_PATH = 'liver_model/test_visualization_data.pkl'

# Functions for preprocessing
def normalize(volume):
    """Normalize volume according to HU windowing and min-max normalization"""
    HU_MIN, HU_MAX = -100, 400
    volume = np.clip(volume, HU_MIN, HU_MAX)
    volume = (volume - HU_MIN) / (HU_MAX - HU_MIN)
    return volume

def resample_volume_and_mask(vol_sitk, target_spacing=[1.5, 1.5, 1.5]):
    """Resample volume to target spacing"""
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

    return vol_resampled


def preprocess_liver_ct(volume_path, target_shape=None):
    """Preprocess the uploaded CT scan"""
    try:
        # Load the volume
        vol_sitk = sitk.ReadImage(volume_path)
       
        # Get original spacing
        original_spacing = vol_sitk.GetSpacing()
       
        # Resample to isotropic voxel spacing
        target_spacing = [1.5, 1.5, 1.5]
        vol_resampled = resample_volume_and_mask(vol_sitk, target_spacing)
       
        # Convert to numpy array
        vol_np = sitk.GetArrayFromImage(vol_resampled).transpose(2, 1, 0)
       
        # Apply noise reduction
        vol_np = gaussian_filter(vol_np, sigma=0.5)
       
        # HU windowing + normalization
        vol_np = normalize(vol_np)
       
        # Pad to target shape if provided
        if target_shape is not None:
            # Calculate padding for each dimension
            pad_d = max(0, target_shape[0] - vol_np.shape[0])
            pad_h = max(0, target_shape[1] - vol_np.shape[1])
            pad_w = max(0, target_shape[2] - vol_np.shape[2])

            # Calculate padding before and after
            pad_d_before = pad_d // 2
            pad_d_after = pad_d - pad_d_before
            pad_h_before = pad_h // 2
            pad_h_after = pad_h - pad_h_before
            pad_w_before = pad_w // 2
            pad_w_after = pad_w - pad_w_before
           
            # Pad the volume
            padding = ((pad_d_before, pad_d_after), (pad_h_before, pad_h_after), (pad_w_before, pad_w_after))
            vol_np = np.pad(vol_np, padding, mode='constant', constant_values=0)
       
        return vol_np, {
            'original_spacing': original_spacing,
            'resampled_shape': vol_np.shape,
            'resampled_spacing': target_spacing
        }
       
    except Exception as e:
        st.error(f"Error in preprocessing: {e}")
        return None, None

def calculate_tumor_severity(mask):
    """Calculate tumor severity score"""
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


def draw_slice_with_tumor_box(volume, tumor_bbox, slice_idx=None):
    """Draw a slice with green bounding box around tumor"""
    if tumor_bbox is None:
        # If no tumor, just show the slice
        if slice_idx is None:
            slice_idx = volume.shape[0] // 2
       
        plt.figure(figsize=(8, 8))
        plt.imshow(volume[slice_idx], cmap='gray')
        plt.title(f"Slice {slice_idx} - No tumor detected")
        plt.axis('off')
       
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        return Image.open(buf)
   
    # Unpack bounding box
    z_min, y_min, x_min, z_max, y_max, x_max = tumor_bbox
   
    # If slice not specified, use middle of tumor
    if slice_idx is None:
        slice_idx = (z_min + z_max) // 2
   
    # Ensure slice is within tumor
    slice_idx = max(z_min, min(slice_idx, z_max))
   
    # Draw the slice with bounding box
    plt.figure(figsize=(8, 8))
    plt.imshow(volume[slice_idx], cmap='gray')
   
    # Draw green rectangle around tumor
    width = x_max - x_min
    height = y_max - y_min
    rect = Rectangle((x_min, y_min), width, height,
                     linewidth=2, edgecolor='lime', facecolor='none')
    plt.gca().add_patch(rect)
   
    plt.title(f"Slice {slice_idx} - Tumor Detected")
    plt.axis('off')
   
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
   
    return Image.open(buf)

# Function to create a multi-slice visualization
def create_multi_slice_visualization(volume, tumor_bbox=None, num_slices=5):
    """Create a grid of slices with tumor bounding boxes"""
    if tumor_bbox is None:
        # If no tumor, show evenly spaced slices
        slice_indices = np.linspace(0, volume.shape[0]-1, num_slices).astype(int)
       
        plt.figure(figsize=(15, 3))
        for i, slice_idx in enumerate(slice_indices):
            plt.subplot(1, num_slices, i+1)
            plt.imshow(volume[slice_idx], cmap='gray')
            plt.title(f"Slice {slice_idx}")
            plt.axis('off')
           
    else:
        # Unpack bounding box
        z_min, y_min, x_min, z_max, y_max, x_max = tumor_bbox
       
        # Calculate slice indices within the tumor with padding
        tumor_depth = z_max - z_min + 1
        padding = max(0, (num_slices - tumor_depth) // 2)
       
        if tumor_depth >= num_slices:
            # If tumor larger than requested slices, sample evenly within tumor
            slice_indices = np.linspace(z_min, z_max, num_slices).astype(int)
        else:
            # If tumor smaller than requested slices, add context before/after
            mid_slices = np.arange(z_min, z_max + 1)
            before_slices = np.arange(max(0, z_min - padding), z_min)
            after_slices = np.arange(z_max + 1, min(volume.shape[0], z_max + 1 + padding))
           
            # Combine slices
            slice_indices = np.concatenate([before_slices, mid_slices, after_slices])
           
            # Ensure we have exactly num_slices by sampling if too many
            if len(slice_indices) > num_slices:
                indices = np.linspace(0, len(slice_indices)-1, num_slices).astype(int)
                slice_indices = slice_indices[indices]
           
            # Pad with more context if too few
            while len(slice_indices) < num_slices:
                if z_min > 0:
                    z_min -= 1
                    slice_indices = np.insert(slice_indices, 0, z_min)
                elif z_max < volume.shape[0] - 1:
                    z_max += 1
                    slice_indices = np.append(slice_indices, z_max)
                   
                if len(slice_indices) < num_slices:
                    continue
       
        # Sort slice indices
        slice_indices = np.sort(slice_indices)
       
        # Draw the slices
        plt.figure(figsize=(15, 3))
        for i, slice_idx in enumerate(slice_indices):
            plt.subplot(1, num_slices, i+1)
            plt.imshow(volume[slice_idx], cmap='gray')
           
            # Draw green rectangle if this slice is within tumor z-bounds
            if z_min <= slice_idx <= z_max:
                width = x_max - x_min
                height = y_max - y_min
                rect = Rectangle((x_min, y_min), width, height,
                                linewidth=2, edgecolor='lime', facecolor='none')
                plt.gca().add_patch(rect)
           
            plt.title(f"Slice {slice_idx}")
            plt.axis('off')
   
    plt.tight_layout()
   
    # Save figure to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
   
    return Image.open(buf)

# Load model
@st.cache_resource
def load_model(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load visualization data for examples
@st.cache_data
def load_visualization_data(viz_data_path):
    try:
        with open(viz_data_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.warning(f"Could not load visualization data: {e}")
        return []

# Main function
def main():
    # Add app title and description
    st.title("?? Liver Tumor Detection System")
    st.markdown("""
    **Detect and analyze liver tumors from CT scans using a 3D CNN model.**
   
    This application allows you to:
    - View example scans with detected tumors
    - Upload your own CT scan for analysis
    - See tumor detection with bounding boxes
    - Get severity assessment based on tumor characteristics
    """)
   
    # Create tabs
    tab1, tab2 = st.tabs(["View Examples", "Upload CT Scan"])
   
    # Example data tab
    with tab1:
        st.header("Example Cases")
       
        # Load visualization data
        viz_data = load_visualization_data(VIZ_DATA_PATH)
       
        if not viz_data:
            st.warning("Example data not available. Please try uploading your own CT scan.")
        else:
            # Create selection for examples
            example_cases = [f"Example {i+1}: {'Tumor' if data.get('actual_label', 0) == 1 else 'No Tumor'}"
                          for i, data in enumerate(viz_data)]
           
            selected_example = st.selectbox("Select an example case:", example_cases)
            example_idx = example_cases.index(selected_example)
           
            # Display the selected example
            example_data = viz_data[example_idx]
            volume = example_data['volume']
           
            # Display results
            col1, col2 = st.columns([2, 1])
           
            with col1:
                # Show multi-slice visualization
                tumor_bbox = example_data.get('tumor_bbox')
                multi_slice_img = create_multi_slice_visualization(volume, tumor_bbox)
                st.image(multi_slice_img, caption="Multiple slices visualization", use_container_width=True)
               
                # Add slider to explore individual slices
                selected_slice = st.slider("Explore slices:", 0, volume.shape[0]-1,
                                        volume.shape[0]//2, key=f"slice_slider_example_{example_idx}")
               
                # Show selected slice with tumor box if present
                slice_img = draw_slice_with_tumor_box(volume, tumor_bbox, selected_slice)
                st.image(slice_img, caption=f"Slice {selected_slice}", use_container_width=True)
           
            with col2:
                # Show prediction info
                st.subheader("Detection Results")
               
                # Display probability
                prob_value = example_data.get('predicted_prob', 0)
                prob_color = "red" if prob_value > 0.5 else "green"
               
                st.markdown(f"""
                **Prediction**: {'Tumor Detected' if prob_value > 0.5 else 'No Tumor'}
               
                **Confidence**: <span style='color:{prob_color};font-weight:bold;'>{prob_value:.1%}</span>
                """, unsafe_allow_html=True)
               
                # Display severity if tumor detected
                if prob_value > 0.5 and 'severity_score' in example_data:
                    severity_score = example_data.get('severity_score', 0)
                    severity_category = example_data.get('severity_category', 'Unknown')
                   
                    # Create a severity gauge
                    st.subheader("Tumor Severity Assessment")
                   
                    # Colored severity indicator
                    if severity_score < 3:
                        severity_color = "green"
                    elif severity_score < 6:
                        severity_color = "orange"
                    else:
                        severity_color = "red"
                   
                    st.markdown(f"""
                    **Severity Score**: <span style='color:{severity_color};font-weight:bold;'>{severity_score}/10</span>
                   
                    **Category**: {severity_category}
                    """, unsafe_allow_html=True)
                   
                    # Visualization of severity score
                    progress_placeholder = st.empty()  
                    progress_placeholder.progress(severity_score / 10)
                else:
                    st.info("No tumor detected in this example.")
   
    # Upload tab
    with tab2:
        st.header("Upload Your CT Scan")
       
        # Load model
        model = load_model(MODEL_PATH)
       
        if model is None:
            st.error("Failed to load model. Please check the model path.")
            return
       
        # Upload NIFTI file
        uploaded_file = st.file_uploader("Upload a CT scan in NIFTI format (.nii or .nii.gz)",
                                        type=["nii", "nii.gz"])
       
        if uploaded_file is not None:
            # Create progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
           
            # Create temporary file to save the upload
            with tempfile.NamedTemporaryFile(delete=False, suffix='.nii') as tmp_file:
                # Write uploaded file to temporary file
                shutil.copyfileobj(uploaded_file, tmp_file)
                tmp_path = tmp_file.name
               
                try:
                    # Update progress
                    status_text.text("Loading and preprocessing CT scan...")
                    progress_bar.progress(0.2)
                   
                    # Get target shape from model's input layer
                    target_shape = model.input_shape[1:4]  # [batch, depth, height, width, channels]
                   
                    # Preprocess CT scan
                    volume, metadata = preprocess_liver_ct(tmp_path, target_shape)
                   
                    if volume is None:
                        st.error("Failed to preprocess the CT scan. Please try another file.")
                        return
                   
                    # Update progress
                    progress_bar.progress(0.5)
                    status_text.text("Running detection model...")
                   
                    # Run prediction
                    # Model expects [batch, depth, height, width, channels]
                    volume_input = np.expand_dims(volume, axis=(0, -1))
                    prediction = model.predict(volume_input)[0][0]
                   
                    # Update progress
                    progress_bar.progress(0.8)
                    status_text.text("Visualizing results...")
                   
                    # Process results
                    is_tumor = prediction > 0.5
                   
                    # Generate approximate tumor bounding box if tumor detected
                    tumor_bbox = None
                    if is_tumor:
                        tumor_bbox = generate_approximate_tumor_bbox(volume, prediction)
                   
                    # Calculate severity
                    severity_score, severity_category = calculate_tumor_severity(volume, prediction)
                   
                    # Complete progress
                    progress_bar.progress(1.0)
                    status_text.text("Analysis complete!")
                    time.sleep(0.5)
                    status_text.empty()
                    progress_bar.empty()
                   
                    # Display results
                    col1, col2 = st.columns([2, 1])
                   
                    with col1:
                        # Show multi-slice visualization
                        multi_slice_img = create_multi_slice_visualization(volume, tumor_bbox)
                        st.image(multi_slice_img, caption="Multiple slices visualization", use_container_width=True)
                       
                        # Add slider to explore individual slices
                        selected_slice = st.slider("Explore slices:", 0, volume.shape[0]-1,
                                                volume.shape[0]//2, key="slice_slider_upload")
                       
                        # Show selected slice with tumor box if present
                        slice_img = draw_slice_with_tumor_box(volume, tumor_bbox, selected_slice)
                        st.image(slice_img, caption=f"Slice {selected_slice}", use_container_width=True)
                   
                    with col2:
                        # Show prediction info
                        st.subheader("Detection Results")
                       
                        # Display probability with colored text
                        prob_color = "red" if prediction > 0.5 else "green"
                       
                        st.markdown(f"""
                        **Prediction**: {'Tumor Detected' if prediction > 0.5 else 'No Tumor'}
                       
                        **Confidence**: <span style='color:{prob_color};font-weight:bold;'>{prediction:.1%}</span>
                        """, unsafe_allow_html=True)
                       
                        # Display severity if tumor detected
                        if prediction > 0.5:
                            # Create a severity gauge
                            st.subheader("Tumor Severity Assessment")
                           
                            # Colored severity indicator
                            if severity_score < 3:
                                severity_color = "green"
                            elif severity_score < 6:
                                severity_color = "orange"
                            else:
                                severity_color = "red"
                           
                            st.markdown(f"""
                            **Severity Score**: <span style='color:{severity_color};font-weight:bold;'>{severity_score}/10</span>
                           
                            **Category**: {severity_category}
                            """, unsafe_allow_html=True)
                           
                            # Visualization of severity score
                            progress_placeholder = st.empty()
                            progress_placeholder.progress(severity_score / 10)
                           
                            # Additional information about the tumor
                            if tumor_bbox:
                                z_min, y_min, x_min, z_max, y_max, x_max = tumor_bbox
                                tumor_size = (z_max - z_min, y_max - y_min, x_max - x_min)
                               
                                st.markdown(f"""
                                **Estimated Tumor Size**:
                                - Depth: {tumor_size[0]} slices
                                - Height: {tumor_size[1]} pixels
                                - Width: {tumor_size[2]} pixels
                                """)
                           
                            # Clinical recommendations
                            st.subheader("Clinical Recommendations")
                            st.markdown("""
                            **Note**: These are automated suggestions and should be reviewed by a medical professional.
                           
                            - Consult with a specialist for a complete diagnosis
                            - Further examination recommended
                            - Consider follow-up imaging in 3-6 months
                            """)
                        else:
                            st.info("No tumor detected in this scan.")
                           
                        # Disclaimer
                        st.caption("""
                        **Disclaimer**: This tool is for research purposes only and is not intended for
                        clinical diagnosis. Consult with healthcare professionals for medical advice.
                        """)
               
                except Exception as e:
                    st.error(f"Error processing the CT scan: {e}")
               
                finally:
                    # Clean up the temporary file
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

if __name__ == "__main__":
    main()
