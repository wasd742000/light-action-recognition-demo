# VideoSage Streamlit App

A web application for video summarization using VideoSage (Graph Neural Networks).

## Features

- **Video Upload**: Support for multiple video formats (MP4, AVI, MOV, MKV, MPG, MPEG)
- **Model Configuration**: Upload custom models and configuration files
- **Adjustable Parameters**: 
  - Keyframe selection percentile
  - Temporal window size
  - Skip factor for graph connections
- **Visual Results**: Display extracted keyframes with importance scores


# Install requirements if not already installed
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py --server.port 5000 --server.address localhost

