# import streamlit as st
# import os
# import tempfile
# import shutil
# import cv2
# import torch
# import numpy as np
# import yaml
# from PIL import Image
# import zipfile
# from io import BytesIO
# import argparse

# # Import your modules
# from videosage.inference import predict_single, get_cfg, build_model
# from videosage.single_graph_generate import generate_video_temporal_graph
# from videosage.single_vidsum import single_vidsum_video, extract_keyframes_adaptive
import streamlit as st
import os
import tempfile
import shutil
import cv2
import torch
import numpy as np
import yaml
from PIL import Image
import zipfile
from io import BytesIO
import argparse

# Import your modules
from videosage.inference import predict_single, get_cfg, build_model
from videosage.single_graph_generate import generate_video_temporal_graph
from videosage.single_vidsum import single_vidsum_video, extract_keyframes_adaptive

# Import action recognition
from inference_actreg import predict_action_from_frames



# Configure Streamlit page
st.set_page_config(
    page_title="Nhận dạng hành động dựa trên tóm tắt nội dung video - Demo",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .default-file-info {
        background-color: #e7f3ff;
        color: #0066cc;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

def load_config_from_file(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        return cfg
    except Exception as e:
        st.error(f"Error loading config file: {str(e)}")
        return None

def create_temp_config(cfg_dict):
    """Create a temporary config file"""
    temp_cfg = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.dump(cfg_dict, temp_cfg, default_flow_style=False)
    temp_cfg.close()
    return temp_cfg.name

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'duration': duration,
        'width': width,
        'height': height
    }

def create_download_zip(keyframes_dir):
    """Create a ZIP file containing all keyframes"""
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for filename in os.listdir(keyframes_dir):
            if filename.endswith('.jpg'):
                file_path = os.path.join(keyframes_dir, filename)
                zip_file.write(file_path, filename)
    
    zip_buffer.seek(0)
    return zip_buffer

def check_default_files():
    """Check if default config and model files exist"""
    config_exists = os.path.exists('cfg.yaml')
    model_exists = os.path.exists('ckpt_best.pt')
    return config_exists, model_exists

def main():
    # Check for default files
    config_exists, model_exists = check_default_files()
    
    # Main header
    st.markdown('<h1 class="main-header">🎬 Nhận dạng hành động dựa trên tóm tắt nội dung video - Demo</h1>', unsafe_allow_html=True)
    # Add this at the end of the sidebar configuration
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 👨‍💻 Author")
    st.sidebar.markdown("""
    **Name:** Vu Hong Phuc 
    **Email:** 18120515@student.hcmus.edu.vn  
    **Institution:** HCMUS  
    **Student ID:** 18120515  
    **Version:** 1.0.0
    """)

    
    # Add this after the video processing parameters section in the sidebar
    st.sidebar.markdown("### Action Recognition")

    enable_action_recognition = st.sidebar.checkbox(
        "Enable Action Recognition",
        value=True,
        help="Predict action from extracted keyframes"
    )

    if enable_action_recognition:
        # Action recognition model file upload
        use_custom_action_model = st.sidebar.checkbox("Use Custom Action Model", value=False)
        
        if use_custom_action_model:
            action_model_file = st.sidebar.file_uploader(
                "Upload Action Model (.pkl)",
                type=['pkl'],
                help="Upload your trained SVM action recognition model"
            )
        else:
            action_model_file = None
            # Check if default action model exists
            if not os.path.exists('svm_model.pkl'):
                st.sidebar.warning("Default action model 'svm_model.pkl' not found")
        
        action_max_frames = st.sidebar.slider(
            "Max Frames for Action Recognition",
            min_value=10,
            max_value=50,
            value=30,
            help="Maximum number of frames to use for action prediction"
        )



    # Sidebar for configuration
    st.sidebar.markdown('<h2 class="sub-header">⚙️ Configuration</h2>', unsafe_allow_html=True)
    
    # Model configuration
    st.sidebar.markdown("### Model Settings")
    
    # Default files status
    if config_exists:
        st.sidebar.markdown('<div class="default-file-info">✅ Default config: cfg.yaml found</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="error-box">❌ Default config: cfg.yaml not found</div>', unsafe_allow_html=True)
    
    if model_exists:
        st.sidebar.markdown('<div class="default-file-info">✅ Default model: ckpt_best.pt found</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="error-box">❌ Default model: ckpt_best.pt not found</div>', unsafe_allow_html=True)
    
    # Model file upload
    use_custom_model = st.sidebar.checkbox("Use Custom Model", value=not model_exists)
    
    if use_custom_model:
        model_file = st.sidebar.file_uploader(
            "Upload Model File (.pth)",
            type=['pth'],
            help="Upload your trained VideoSage model file"
        )
    else:
        model_file = None
        if not model_exists:
            st.sidebar.error("Default model file 'ckpt_best.pt' not found. Please upload a custom model.")
    
    # Configuration file upload
    use_custom_config = st.sidebar.checkbox("Use Custom Configuration", value=not config_exists)
    
    if use_custom_config:
        config_file = st.sidebar.file_uploader(
            "Upload Config File (.yaml)",
            type=['yaml', 'yml'],
            help="Upload your model configuration file"
        )
    else:
        config_file = None
        if not config_exists:
            st.sidebar.error("Default config file 'cfg.yaml' not found. Please upload a custom config.")
    
    # Display current configuration if using default
    if not use_custom_config and config_exists:
        with st.sidebar.expander("View Current Configuration"):
            cfg = load_config_from_file('cfg.yaml')
            if cfg:
                st.json(cfg)
    
    # Video processing parameters
    st.sidebar.markdown("### Processing Parameters")
    
    percentile = st.sidebar.slider(
        "Keyframe Selection Percentile",
        min_value=50,
        max_value=99,
        value=90,
        help="Higher values = fewer but more important keyframes"
    )
    
    tauf = st.sidebar.slider(
        "Temporal Window (tauf)",
        min_value=5,
        max_value=50,
        value=10,
        help="Maximum frame difference for graph connections"
    )
    
    skip_factor = st.sidebar.slider(
        "Skip Factor",
        min_value=0,
        max_value=20,
        value=0,
        help="Additional connections between non-adjacent nodes"
    )
    
    single_highest_frame = st.sidebar.checkbox(
        "Extract Only Highest Scoring Frame",
        value=False,
        help="Extract only the single most important frame"
    )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">📹 Video Upload</h2>', unsafe_allow_html=True)
        
        # Video upload
        uploaded_video = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv', 'mpg', 'mpeg'],
            help="Upload the video you want to summarize"
        )
        
        if uploaded_video is not None:
            # Display video information
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                tmp_video.write(uploaded_video.read())
                tmp_video_path = tmp_video.name
            
            video_info = get_video_info(tmp_video_path)
            
            if video_info:
                st.markdown('<div class="info-box">', unsafe_allow_html=True)
                st.markdown("**Video Information:**")
                st.write(f"• Duration: {video_info['duration']:.2f} seconds")
                st.write(f"• Frame Count: {video_info['frame_count']}")
                st.write(f"• FPS: {video_info['fps']:.2f}")
                st.write(f"• Resolution: {video_info['width']}x{video_info['height']}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Display video
                st.video(uploaded_video)
            
            # Clean up temporary file
            os.unlink(tmp_video_path)
    
    with col2:
        st.markdown('<h2 class="sub-header">🔍 Processing Status</h2>', unsafe_allow_html=True)
        
        # Processing status area
        status_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        # Replace the existing processing section with this updated version
    if st.button("🚀 Start Video Summarization", type="primary", use_container_width=True):
        if uploaded_video is None:
            st.error("Please upload a video file first!")
            return
        
        # Check model availability
        if use_custom_model:
            if model_file is None:
                st.error("Please upload a model file!")
                return
        else:
            if not model_exists:
                st.error("Default model file 'ckpt_best.pt' not found!")
                return
        
        # Check config availability
        if use_custom_config:
            if config_file is None:
                st.error("Please upload a config file!")
                return
        else:
            if not config_exists:
                st.error("Default config file 'cfg.yaml' not found!")
                return
        
        # Check action recognition model if enabled
        if enable_action_recognition:
            if use_custom_action_model:
                if action_model_file is None:
                    st.error("Please upload an action recognition model!")
                    return
            else:
                if not os.path.exists('svm_model.pkl'):
                    st.error("Default action model 'svm_model.pkl' not found!")
                    return
        
        try:
            # Create temporary directories
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded video
                video_path = os.path.join(temp_dir, "input_video.mp4")
                with open(video_path, "wb") as f:
                    f.write(uploaded_video.getvalue())
                
                # Handle model file
                if use_custom_model:
                    model_path = os.path.join(temp_dir, "model.pth")
                    with open(model_path, "wb") as f:
                        f.write(model_file.getvalue())
                else:
                    model_path = "ckpt_best.pt"  # Use default model
                
                # Handle configuration file
                if use_custom_config:
                    config_path = os.path.join(temp_dir, "config.yaml")
                    with open(config_path, "wb") as f:
                        f.write(config_file.getvalue())
                else:
                    config_path = "cfg.yaml"  # Use default config
                
                # Handle action recognition model
                if enable_action_recognition:
                    if use_custom_action_model:
                        action_model_path = os.path.join(temp_dir, "action_model.pkl")
                        with open(action_model_path, "wb") as f:
                            f.write(action_model_file.getvalue())
                    else:
                        action_model_path = "svm_model.pkl"  # Use default action model
                
                keyframes_dir = os.path.join(temp_dir, "keyframes")
                
                # Update status
                status_placeholder.info("🔄 Processing video...")
                progress_bar = progress_placeholder.progress(0)
                
                # Step 1: Generate graph
                progress_bar.progress(20)
                status_placeholder.info("📊 Generating temporal graph...")
                
                # Step 2: Run inference
                progress_bar.progress(40)
                status_placeholder.info("🧠 Running model inference...")
                
                # Step 3: Extract keyframes
                progress_bar.progress(60)
                status_placeholder.info("🖼️ Extracting keyframes...")
                
                # Process video
                single_vidsum_video(
                    input_video=video_path,
                    output_dir=keyframes_dir,
                    model_path=model_path,
                    cfg_path=config_path,
                    percentile=percentile,
                    tauf=tauf,
                    skip_factor=skip_factor,
                    enable_single_highest_frame=single_highest_frame
                )
                
                # Step 4: Action recognition (if enabled)
                predicted_action = None
                confidence_scores = None
                
                if enable_action_recognition:
                    progress_bar.progress(80)
                    status_placeholder.info("🎭 Recognizing action from keyframes...")
                    
                    try:
                        predicted_action, confidence_scores = predict_action_from_frames(
                            frame_folder=keyframes_dir,
                            model_path=action_model_path,
                            max_frames=action_max_frames,
                            batch_size=32
                        )
                    except Exception as e:
                        st.warning(f"Action recognition failed: {str(e)}")
                        predicted_action = None
                        confidence_scores = None
                
                progress_bar.progress(100)
                status_placeholder.success("✅ Processing completed!")
                
                # Display results
                st.markdown('<h2 class="sub-header">📸 Results</h2>', unsafe_allow_html=True)
                
                # Display action recognition results if available
                if enable_action_recognition and predicted_action:
                    st.markdown('<h3 class="sub-header">🎭 Action Recognition</h3>', unsafe_allow_html=True)
                    
                    col_action1, col_action2 = st.columns([1, 1])
                    
                    with col_action1:
                        st.markdown(f'<div class="success-box"><h4>Predicted Action: {predicted_action}</h4></div>', unsafe_allow_html=True)
                    
                    with col_action2:
                        if confidence_scores:
                            st.markdown("**Confidence Scores:**")
                            # Sort by confidence and display top 5
                            sorted_scores = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)[:5]
                            for action, score in sorted_scores:
                                st.write(f"• {action}: {score:.4f}")
                
                # Display keyframes
                st.markdown('<h3 class="sub-header">📸 Extracted Keyframes</h3>', unsafe_allow_html=True)
                
                if os.path.exists(keyframes_dir):
                    keyframe_files = [f for f in os.listdir(keyframes_dir) if f.endswith('.jpg')]
                    keyframe_files.sort()
                    
                    if keyframe_files:
                        st.markdown(f'<div class="success-box">Successfully extracted {len(keyframe_files)} keyframes!</div>', unsafe_allow_html=True)
                        
                        # Create download button
                        zip_buffer = create_download_zip(keyframes_dir)
                        st.download_button(
                            label="📥 Download All Keyframes (ZIP)",
                            data=zip_buffer.getvalue(),
                            file_name="keyframes.zip",
                            mime="application/zip",
                            use_container_width=True
                        )
                        
                        # Display keyframes in a grid
                        cols_per_row = 3
                        for i in range(0, len(keyframe_files), cols_per_row):
                            cols = st.columns(cols_per_row)
                            for j, col in enumerate(cols):
                                if i + j < len(keyframe_files):
                                    keyframe_file = keyframe_files[i + j]
                                    keyframe_path = os.path.join(keyframes_dir, keyframe_file)
                                    
                                    # Extract score from filename
                                    try:
                                        score = keyframe_file.split('_score_')[1].split('.jpg')[0]
                                        frame_num = keyframe_file.split('_')[1]
                                    except:
                                        score = "N/A"
                                        frame_num = str(i + j)
                                    
                                    with col:
                                        image = Image.open(keyframe_path)
                                        caption = f"Frame {frame_num} (Score: {score})"
                                        if enable_action_recognition and predicted_action:
                                            caption += f"\nAction: {predicted_action}"
                                        st.image(
                                            image,
                                            caption=caption,
                                            use_container_width=True
                                        )
                    else:
                        st.warning("No keyframes were extracted. Try adjusting the percentile threshold.")
                        
        except Exception as e:
            status_placeholder.error(f"❌ Error during processing: {str(e)}")
            st.exception(e)

    
    # Footer with file requirements
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666;">
            <p><strong>VideoSage - Video Summarization using Graph Neural Networks</strong></p>
            <p>Default files: <code>cfg.yaml</code> (config) and <code>ckpt_best.pt</code> (model)</p>
            <p>Upload a video to extract the most important frames automatically</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Instructions section
    # Update the instructions section to include action recognition info
    with st.expander("📋 Instructions & Requirements"):
        st.markdown("""
        ### Default Files Required:
        - **cfg.yaml**: Model configuration file (should be in the same directory as app.py)
        - **ckpt_best.pt**: Pre-trained model weights (should be in the same directory as app.py)
        - **svm_model.pkl**: Action recognition model (optional, for action recognition feature)
        
        ### How to Use:
        1. **Ensure default files exist** or upload custom ones using the sidebar options
        2. **Upload a video** in supported formats (MP4, AVI, MOV, MKV, MPG, MPEG)
        3. **Adjust parameters** in the sidebar:
        - **Percentile**: Higher values = fewer keyframes (more selective)
        - **Temporal Window (tauf)**: Controls graph connectivity
        - **Skip Factor**: Additional graph connections
        - **Single Highest Frame**: Extract only the most important frame
        - **Enable Action Recognition**: Predict action from extracted keyframes
        4. **Click "Start Video Summarization"** to process
        5. **View results**: Action prediction and keyframes
        6. **Download results** as individual images or ZIP file
        
        ### Features:
        - **Video Summarization**: Extract key frames using VideoSage
        - **Action Recognition**: Predict action from summarized frames using CLIP + SVM
        - **Batch Processing**: Efficient GPU-accelerated processing
        - **Confidence Scores**: View prediction confidence for actions
        
        ### Supported Video Formats:
        - MP4, AVI, MOV, MKV, MPG, MPEG
        
        ### Output:
        - Predicted action with confidence scores
        - Keyframes saved as JPG images
        - Filenames include frame number and importance score
        - ZIP download available for all keyframes
        """)

    
    # Technical details section
    with st.expander("🔧 Technical Details"):
        st.markdown("""
        ### Model Architecture:
        - **SPELL**: Spatial-Temporal Graph Neural Network
        - **Features**: ResNet50-based visual features
        - **Graph**: Temporal connections between video frames
        
        ### Processing Pipeline:
        1. **Feature Extraction**: Extract visual features from video frames
        2. **Graph Generation**: Create temporal graph with configurable connections
        3. **Model Inference**: Run SPELL model to score frame importance
        4. **Keyframe Selection**: Extract frames above adaptive threshold
        
        ### Configuration Parameters:
        - **model_name**: Model architecture (SPELL)
        - **use_spf**: Use spatial features
        - **use_ref**: Use iterative refinement
        - **num_modality**: Number of input modalities
        - **channel1/channel2**: Network layer dimensions
        - **dropout**: Dropout rate for regularization
        """)

if __name__ == "__main__":
    main()

