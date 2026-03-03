"""
Diabetic Retinopathy Classification - Streamlit Web Application
Production-ready UI for DR inference, batch processing, and evaluation.
PyTorch backend with 5 CAM explainability methods.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Import our modules
from src.config import Config
from src.inference import predict_single_image, predict_batch_from_zip
from src.model import DRClassifier
from src.preprocessing import load_image_safe
from src.utils.validation import validate_zip_file

# =====================
# PAGE CONFIG
# =====================
st.set_page_config(
    page_title="DR Classifier",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =====================
# LOAD CONFIG & MODEL (CACHED)
# =====================
@st.cache_resource
def load_config():
    """Load configuration (cached)."""
    try:
        return Config()
    except Exception as e:
        st.error(f"❌ Configuration Error: {e}")
        st.stop()

@st.cache_resource
def load_model(_config):
    """Load model (cached). Underscore prefix prevents hashing."""
    try:
        with st.spinner("Loading model..."):
            model = DRClassifier(_config.MODEL_PATH)
        st.success("✅ Model loaded successfully (PyTorch)")
        return model
    except Exception as e:
        st.error(f"❌ Model Loading Error: {e}")
        st.stop()

config = load_config()
model = load_model(config)

# =====================
# HEADER
# =====================
st.title("🩺 Diabetic Retinopathy Classifier")
st.caption(
    "Production-ready AI system for DR classification with explainability. "
    "**5-class classification:** No DR, Mild, Moderate, Severe, PDR"
)

# =====================
# TABS
# =====================
tab_single, tab_batch, tab_eval, tab_about = st.tabs([
    "📸 Single Image",
    "📦 Batch Inference",
    "📊 Evaluation",
    "ℹ️ About"
])

# =====================
# TAB 1: SINGLE IMAGE
# =====================
with tab_single:
    st.header("Single Image Inference")
    st.write("Upload a retinal fundus image for DR classification with CAM explainability.")

    col1, col2 = st.columns([1, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload Retinal Image",
            type=["jpg", "jpeg", "png"],
            help="Accepted formats: JPG, PNG"
        )

    with col2:
        # CAM method selector
        cam_options = ["None"] + config.CAM_METHODS
        cam_method_choice = st.selectbox(
            "🔬 Explainability Method",
            options=cam_options,
            index=cam_options.index(config.DEFAULT_CAM_METHOD),
            help=(
                "**GradCAM** – Classic gradient-weighted maps\n\n"
                "**GradCAM++** – Better localization for multiple objects\n\n"
                "**ScoreCAM** – Gradient-free, perturbation-based\n\n"
                "**EigenCAM** – Fast PCA of feature maps\n\n"
                "**LayerCAM** – Fine-grained spatial heatmaps"
            )
        )
        show_cam = cam_method_choice != "None"

        top_k = st.slider("Show Top-K Predictions", min_value=1, max_value=5, value=3)
        confidence_threshold = st.slider(
            "Confidence Threshold (for review flag)",
            min_value=0.1,
            max_value=0.95,
            value=float(config.CONFIDENCE_THRESHOLD),
            step=0.05,
            help="Predictions below this threshold will be flagged for review"
        )

    if uploaded_file is not None:
        # Create temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        # Load and display original image
        img_pil = load_image_safe(tmp_path)

        if img_pil is None:
            st.error("❌ Failed to load image. Please try another file.")
        else:
            col_img, col_viz = st.columns([1, 1])

            with col_img:
                st.subheader("📷 Uploaded Image")
                st.image(img_pil, use_container_width=True)

            # Run inference
            with st.spinner("Running inference..."):
                result = predict_single_image(
                    tmp_path,
                    model,
                    config,
                    cam_method=cam_method_choice if show_cam else None
                )

            if not result['success']:
                st.error(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")
            else:
                # Quality warnings
                quality_issues = result.get('quality_issues', [])
                if quality_issues:
                    st.warning(f"⚠️ **Image Quality Issues:** {', '.join(quality_issues)}")

                # Prediction result
                predicted_class = result['class_name']
                confidence = result['confidence']

                # Needs review flag
                needs_review = confidence < confidence_threshold or len(quality_issues) > 0

                if needs_review:
                    st.error("⚠️ **NEEDS REVIEW** - Low confidence or quality issues detected")
                else:
                    st.success("✅ **High Confidence Prediction**")

                # Main prediction
                st.markdown(f"### 🎯 Prediction: **{predicted_class}**")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                # Probability distribution
                st.subheader("📊 Probability Distribution")
                prob_df = pd.DataFrame([
                    {'Class': k, 'Probability': v}
                    for k, v in result['probabilities'].items()
                ]).sort_values('Probability', ascending=False)

                st.bar_chart(prob_df.set_index('Class'), height=250)

                # Top-K table
                st.subheader(f"🏅 Top-{top_k} Predictions")
                st.dataframe(
                    prob_df.head(top_k).style.format({'Probability': '{:.2%}'}),
                    use_container_width=True,
                    hide_index=True
                )

                # CAM visualization
                with col_viz:
                    cam_heatmap = result.get('cam_heatmap')
                    used_method = result.get('cam_method', cam_method_choice)

                    if show_cam and cam_heatmap is not None:
                        st.subheader(f"🔥 {used_method} Explanation")
                        fig = model.create_cam_figure(
                            img_pil,
                            cam_heatmap,
                            method=used_method,
                            alpha=config.CAM_ALPHA
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                        method_descriptions = {
                            "GradCAM":   "Highlights regions using gradient flow through the last conv layer.",
                            "GradCAM++": "Enhanced version with better multi-object localization.",
                            "ScoreCAM":  "Gradient-free — uses activation score perturbations for faithfulness.",
                            "EigenCAM":  "Uses PCA of feature maps — fast and class-agnostic.",
                            "LayerCAM":  "Fine-grained spatial heatmaps with higher resolution detail.",
                        }
                        st.caption(method_descriptions.get(used_method, ""))

                    elif show_cam:
                        st.info(f"⚠️ {used_method} visualization unavailable for this image.")

                # Quality metrics (expandable)
                with st.expander("🔍 Image Quality Metrics"):
                    quality_metrics = result.get('quality_metrics', {})
                    col_q1, col_q2, col_q3 = st.columns(3)

                    with col_q1:
                        st.metric(
                            "Resolution",
                            f"{quality_metrics.get('width', 0)}×{quality_metrics.get('height', 0)}"
                        )

                    with col_q2:
                        blur_score = quality_metrics.get('blur_score', 0)
                        st.metric(
                            "Blur Score",
                            f"{blur_score:.1f}",
                            help="Higher is sharper. < 100 is blurry."
                        )

                    with col_q3:
                        brightness = quality_metrics.get('mean_brightness', 0)
                        st.metric("Mean Brightness", f"{brightness:.1f}")

    else:
        st.info("👆 Upload an image to start inference")

# =====================
# TAB 2: BATCH INFERENCE
# =====================
with tab_batch:
    st.header("Batch Inference from ZIP")
    st.write(
        "Upload a ZIP file containing multiple retinal images. "
        "The system will process all images and generate a downloadable CSV report."
    )

    col_b1, col_b2 = st.columns([2, 1])

    with col_b1:
        uploaded_zip = st.file_uploader(
            "Upload ZIP File",
            type=["zip"],
            help="ZIP should contain retinal images (JPG, PNG)"
        )

    with col_b2:
        batch_confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.95,
            value=float(config.CONFIDENCE_THRESHOLD),
            step=0.05,
            key="batch_confidence"
        )

    if uploaded_zip is not None:
        # Save ZIP temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_zip:
            tmp_zip.write(uploaded_zip.getbuffer())
            tmp_zip_path = tmp_zip.name

        # Validate ZIP
        is_valid, error_msg, num_images = validate_zip_file(tmp_zip_path)

        if not is_valid:
            st.error(f"❌ Invalid ZIP file: {error_msg}")
        elif num_images == 0:
            st.warning("⚠️ ZIP contains no valid image files")
        else:
            st.success(f"✅ Found {num_images} images in ZIP")

            if st.button("🚀 Run Batch Inference", type="primary"):
                st.subheader("Processing...")

                # Progress bar
                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def update_progress(progress):
                    progress_bar.progress(progress)
                    status_text.text(f"Processing: {int(progress * 100)}% complete")

                # Run batch inference
                start_time = time.time()

                with st.spinner("Running batch inference..."):
                    results_df = predict_batch_from_zip(
                        tmp_zip_path,
                        model,
                        config,
                        progress_callback=update_progress
                    )

                duration = time.time() - start_time

                if results_df.empty:
                    st.error("❌ No results generated. All images may have failed.")
                else:
                    total = len(results_df)
                    successful = len(results_df[results_df['predicted_class'] != -1])
                    errors = total - successful
                    needs_review_count = len(results_df[results_df['needs_review'] == True])

                    st.success(f"✅ Batch processing complete in {duration:.2f}s")

                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    with col_m1:
                        st.metric("Total Images", total)
                    with col_m2:
                        st.metric("Successful", successful)
                    with col_m3:
                        st.metric("Errors", errors)
                    with col_m4:
                        st.metric("Needs Review", needs_review_count)

                    st.subheader("📄 Results Preview (First 20 rows)")
                    st.dataframe(results_df.head(20), use_container_width=True, height=400)

                    csv = results_df.to_csv(index=False).encode('utf-8')
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="⬇️ Download Full Results (CSV)",
                        data=csv,
                        file_name=f"dr_batch_results_{timestamp}.csv",
                        mime="text/csv",
                        type="primary"
                    )

                    if successful > 0:
                        with st.expander("📊 Prediction Distribution"):
                            class_counts = results_df[results_df['predicted_class'] != -1]['class_name'].value_counts()
                            st.bar_chart(class_counts)

    else:
        st.info("👆 Upload a ZIP file to start batch processing")

# =====================
# TAB 3: EVALUATION
# =====================
with tab_eval:
    st.header("Model Evaluation Dashboard")
    st.write(
        "View evaluation metrics from the test dataset. "
        "Displays confusion matrix and classification report."
    )

    eval_dir = config.EVAL_DIR
    confusion_matrix_path = eval_dir / "confusion_matrix.png"
    classification_report_path = eval_dir / "classification_report.txt"

    if confusion_matrix_path.exists() and classification_report_path.exists():
        col_e1, col_e2 = st.columns([1, 1])

        with col_e1:
            st.subheader("📊 Confusion Matrix")
            st.image(str(confusion_matrix_path), use_container_width=True)

        with col_e2:
            st.subheader("📄 Classification Report")
            with open(classification_report_path, 'r') as f:
                report_text = f.read()
            st.text(report_text)
    else:
        st.warning(
            "⚠️ Evaluation results not found. "
            "Run evaluation first using the evaluation script."
        )
        st.code("python -m src.evaluation", language="bash")
        st.info(
            "💡 To generate evaluation metrics, you need:\n"
            "- A CSV file with ground truth labels\n"
            "- A directory with test images\n"
            "- Run the evaluation module"
        )

# =====================
# TAB 4: ABOUT
# =====================
with tab_about:
    st.header("About This System")

    st.markdown("""
    ### 🩺 Diabetic Retinopathy Classification System

    Production-ready AI system for automated Diabetic Retinopathy (DR) screening
    from retinal fundus images using deep learning.

    #### 🎯 Features

    - **5-Class Classification:** No DR, Mild, Moderate, Severe, Proliferative DR (PDR)
    - **Single Image Inference:** Upload and analyze individual images
    - **Batch Processing:** Process hundreds of images from ZIP files
    - **5 CAM Explainability Methods:** Visual explanations of model decisions
    - **Quality Checks:** Automatic detection of blur and image quality issues
    - **Confidence Scoring:** Flag low-confidence predictions for expert review
    - **Production Architecture:** Modular, tested, deployable code

    #### 🤖 Model Information

    - **Architecture:** MobileNetV2 (fine-tuned)
    - **Input Size:** 224×224 pixels
    - **Classes:** 5 (No DR, Mild, Moderate, Severe, PDR)
    - **Framework:** PyTorch + torchvision
    - **Deployment:** CPU-optimized

    #### 🔬 CAM Explainability Methods

    | Method | Description | Speed |
    |---|---|---|
    | **GradCAM** | Gradient-weighted class activation maps | ⚡ Fast |
    | **GradCAM++** | Enhanced localization, handles multiple instances | ⚡ Fast |
    | **ScoreCAM** | Gradient-free, perturbation-based (most faithful) | 🐢 Slow |
    | **EigenCAM** | PCA of feature maps — no gradients needed | ⚡ Fast |
    | **LayerCAM** | Fine-grained spatial heatmaps | ⚡ Fast |

    #### 📊 Performance Considerations

    - **Blur Detection:** Images with Laplacian variance < 100 flagged as blurry
    - **Resolution Check:** Minimum 256×256 recommended
    - **Confidence Threshold:** Default 60% (adjustable)
    - **Batch Size:** 16 images per batch (memory-efficient)
    - **ScoreCAM** is significantly slower than others — use for critical cases

    #### ⚙️ Technical Stack

    - **Backend:** Python, PyTorch, torchvision, pytorch-grad-cam
    - **UI:** Streamlit
    - **Image Processing:** PIL, OpenCV
    - **Metrics:** Scikit-learn
    """)

    st.divider()

    st.markdown("""
    ### ⚠️ IMPORTANT MEDICAL DISCLAIMER

    > **This tool is for EDUCATIONAL, RESEARCH, and DEMONSTRATION purposes only.**
    >
    > **NOT FOR CLINICAL USE**
    >
    > This system is:
    > - ❌ NOT FDA approved or clinically validated
    > - ❌ NOT a substitute for professional medical examination
    > - ❌ NOT intended for diagnosis or treatment decisions
    > - ✅ Intended for screening support and research only
    >
    > **Always consult qualified ophthalmologists and healthcare professionals for medical decisions.**
    """)

    st.divider()

    st.markdown("""
    ### 📚 Usage Instructions

    **Single Image Mode:**
    1. Upload a retinal fundus image (JPG/PNG)
    2. Select an explainability method from the dropdown
    3. Adjust confidence threshold if needed
    4. View prediction, probability distribution, and CAM heatmap

    **Batch Mode:**
    1. Prepare a ZIP file with retinal images
    2. Upload the ZIP file
    3. Click "Run Batch Inference"
    4. Download CSV results

    ### 🔧 Configuration

    System configuration can be customized via environment variables or `.env` file:

    - `MODEL_PATH` - Path to .pth model file
    - `CONFIDENCE_THRESHOLD` - Default confidence threshold
    - `BLUR_THRESHOLD` - Blur detection threshold
    - `BATCH_SIZE` - Batch processing size

    ### 📦 Project Structure

    ```
    ├── src/
    │   ├── config.py           # Configuration
    │   ├── model.py            # Model & 5 CAM methods (PyTorch)
    │   ├── preprocessing.py    # Image preprocessing
    │   ├── inference.py        # Inference logic
    │   ├── evaluation.py       # Metrics & evaluation
    │   └── utils/             # Utilities
    ├── app.py                  # Streamlit UI (this file)
    ├── pretrained/             # .pth model files
    ├── outputs/                # Logs & results
    └── requirements.txt        # Dependencies
    ```
    """)

# =====================
# FOOTER
# =====================
st.divider()
st.caption(
    "⚠️ **Medical Disclaimer:** This tool is for educational and research purposes only. "
    "Not for clinical diagnosis. Always consult qualified healthcare professionals."
)
st.caption(f"🤖 Model: {Path(config.MODEL_PATH).name} | 🔥 Framework: PyTorch | 📦 Version: 2.0.0")
