import logging
import os
import time
import datetime
import torch
import joblib
import open_clip
import numpy as np
from PIL import Image


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('action_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add these constants if they're not already defined
network_model_type = "ViT-B-32"  # or whatever CLIP model you're using
CLASSIFIER_TYPE = "LinearSVC"  # or your preferred classifier type

def extract_video_features(frame_folder, model, preprocess, device, max_frames=30, batch_size=32):
    """Extract feature vector by averaging frame embeddings with GPU batch processing."""
    import numpy as np
    from PIL import Image
    
    # Get all image paths - use list comprehension for speed
    image_paths = [
        os.path.join(frame_folder, filename) 
        for filename in sorted(os.listdir(frame_folder))
        if filename.lower().endswith(('png', 'jpg', 'jpeg'))
    ]
    
    total_frames = len(image_paths)
    
    # Limit the number of frames to process if there are too many
    if len(image_paths) > max_frames:
        # Use uniform sampling instead of step-based sampling for better representation
        indices = np.linspace(0, len(image_paths) - 1, max_frames, dtype=int)
        image_paths = [image_paths[i] for i in indices]
    
    if not image_paths:
        return None
    
    logger.info(f"Processing {len(image_paths)} frames from {total_frames} total frames")
    
    all_features = []
    
    # Process frames in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        # Load and preprocess batch of images
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image = preprocess(image)
                batch_images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        # Stack images into batch tensor
        batch_tensor = torch.stack(batch_images).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.encode_image(batch_tensor)
            features = features.cpu().numpy()
            all_features.append(features)
    
    if not all_features:
        logger.error("No features extracted from any frames")
        return None
    
    # Concatenate all features and compute mean
    all_features = np.concatenate(all_features, axis=0)
    video_features = np.mean(all_features, axis=0)
    
    logger.info(f"Extracted feature vector of shape: {video_features.shape}")
    return video_features


def load_trained_model(model_path='svm_model.pkl'):
    """Load the trained SVM model and action classes."""
    try:
        model_data = joblib.load(model_path)
        clf = model_data['model']
        action_classes = model_data['classes']
        logger.info(f"Loaded model with {len(action_classes)} action classes: {action_classes}")
        return clf, action_classes
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def predict_action_from_frames(frame_folder, model_path='svm_model.pkl', max_frames=30, batch_size=32):
    """
    Predict action from a folder of frames using the trained SVM model.
    
    Args:
        frame_folder (str): Path to folder containing video frames
        model_path (str): Path to the saved SVM model
        max_frames (int): Maximum number of frames to process
        batch_size (int): Batch size for processing frames
    
    Returns:
        tuple: (predicted_action, confidence_scores)
    """
    # Configure device and load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load CLIP model (same as training)
    logger.info("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(network_model_type, pretrained='openai')
    model = model.to(device)
    model.eval()
    
    # Load trained SVM model
    logger.info("Loading trained SVM model...")
    clf, action_classes = load_trained_model(model_path)
    
    # Extract features from frames
    logger.info(f"Extracting features from frames in: {frame_folder}")
    features = extract_video_features(frame_folder, model, preprocess, device, max_frames, batch_size)
    
    if features is None:
        logger.error("No features extracted from frames")
        return None, None
    
    # Reshape features for prediction (SVM expects 2D array)
    features = features.reshape(1, -1)
    
    # Make prediction
    logger.info("Making prediction...")
    prediction = clf.predict(features)[0]
    predicted_action = action_classes[prediction]
    
    # Get confidence scores if available
    confidence_scores = None
    if hasattr(clf, 'decision_function'):
        # For SVM, decision_function gives distances to separating hyperplane
        decision_scores = clf.decision_function(features)[0]
        confidence_scores = dict(zip(action_classes, decision_scores))
    elif hasattr(clf, 'predict_proba'):
        # For classifiers that support probability prediction
        probabilities = clf.predict_proba(features)[0]
        confidence_scores = dict(zip(action_classes, probabilities))
    
    logger.info(f"Predicted action: {predicted_action}")
    if confidence_scores:
        logger.info("Confidence scores:")
        for action, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {action}: {score:.4f}")
    
    return predicted_action, confidence_scores

def inference_main(frame_folder, model_path='svm_model.pkl', max_frames=30, batch_size=32, log_level='INFO'):
    """
    Main function for inference on a folder of frames.
    
    Args:
        frame_folder (str): Path to folder containing video frames
        model_path (str): Path to the saved SVM model
        max_frames (int): Maximum number of frames to process
        batch_size (int): Batch size for processing frames
        log_level (str): Logging level
    """
    # Set log level
    logger.setLevel(getattr(logging, log_level))
    
    # Configure GPU settings (same as training)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    try:
        predicted_action, confidence_scores = predict_action_from_frames(
            frame_folder, model_path, max_frames, batch_size
        )
        
        if predicted_action:
            print(f"\nPredicted Action: {predicted_action}")
            
            if confidence_scores:
                print("\nConfidence Scores:")
                for action, score in sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {action}: {score:.4f}")
        else:
            print("Failed to predict action")
            
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        print(f"Error: {e}")

# Example usage function
def batch_inference(video_folders, model_path='svm_model.pkl', max_frames=30, batch_size=32):
    """
    Perform inference on multiple video folders.
    
    Args:
        video_folders (list): List of paths to folders containing video frames
        model_path (str): Path to the saved SVM model
        max_frames (int): Maximum number of frames to process per video
        batch_size (int): Batch size for processing frames
    
    Returns:
        dict: Dictionary mapping folder paths to predicted actions
    """
    results = {}
    
    # Load model once for all predictions
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load(network_model_type, device=device, jit=True)
    model.eval()
    clf, action_classes = load_trained_model(model_path)
    
    for folder in video_folders:
        logger.info(f"Processing folder: {folder}")
        
        # Extract features
        features = extract_video_features(folder, model, preprocess, device, max_frames, batch_size)
        
        if features is not None:
            # Make prediction
            features = features.reshape(1, -1)
            prediction = clf.predict(features)[0]
            predicted_action = action_classes[prediction]
            results[folder] = predicted_action
            logger.info(f"Folder {folder}: {predicted_action}")
        else:
            results[folder] = "ERROR: No features extracted"
            logger.error(f"Failed to extract features from {folder}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Action Recognition Inference')
    parser.add_argument('--mode', choices=['train', 'inference', 'batch_inference'], 
                       default='train', help='Mode: train or inference')
    parser.add_argument('--frame_folder', type=str, help='Path to folder containing frames for inference')
    parser.add_argument('--model_path', type=str, default='svm_model.pkl', 
                       help='Path to saved SVM model')
    parser.add_argument('--dataset', type=str, help='Dataset path for training')
    parser.add_argument('--max_frames', type=int, default=30, help='Maximum frames to process')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Your existing training code
        main(
            dataset=args.dataset,
            max_frames=args.max_frames,
            batch_size=args.batch_size,
            log_level=args.log_level
        )
    elif args.mode == 'inference':
        # Single folder inference
        inference_main(
            frame_folder=args.frame_folder,
            model_path=args.model_path,
            max_frames=args.max_frames,
            batch_size=args.batch_size,
            log_level=args.log_level
        )
    elif args.mode == 'batch_inference':
        # Multiple folders inference
        video_folders = [args.frame_folder]  # You can extend this to multiple folders
        results = batch_inference(
            video_folders=video_folders,
            model_path=args.model_path,
            max_frames=args.max_frames,
            batch_size=args.batch_size
        )
        print("Batch inference results:")
        for folder, action in results.items():
            print(f"  {folder}: {action}")
