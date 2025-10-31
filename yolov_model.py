import cv2
import base64
import numpy as np
import torch
from pathlib import Path
import io
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model ONCE
MODEL_PATH = Path(__file__).parent / "best.pt"
model = None

# Define the label swap mapping
LABEL_SWAP = {
    "caries": "calculus",
    "calculus": "caries"
}

def load_model():
    """
    Load YOLOv5 model and swap the class names
    """
    global model
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=str(MODEL_PATH))
        
        # Set model parameters
        model.conf = 0.25  # Confidence threshold
        model.iou = 0.45   # NMS IoU threshold
        
        # Swap the class names in the model
        original_names = model.names.copy()
        for class_id, class_name in original_names.items():
            if class_name.lower() in LABEL_SWAP:
                model.names[class_id] = LABEL_SWAP[class_name.lower()]
        
        logger.info(f"YOLOv5 model loaded successfully")
        logger.info(f"Class names after swap: {model.names}")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise

# Load the model when module is imported
load_model()

def predict(image_bytes):
    """
    Perform prediction using YOLOv5 model
    
    Args:
        image_bytes: Image data in bytes
        
    Returns:
        Dictionary with detections and base64 encoded image
    """
    try:
        # Convert bytes to PIL Image
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert PIL Image to numpy array for OpenCV
        img_np = np.array(image)
        
        # Run YOLOv5 prediction
        results = model(img_np)
        
        # Process the results - labels are already swapped via model.names
        detections = []
        predictions = results.pandas().xyxy[0]
        
        for _, detection in predictions.iterrows():
            detections.append({
                "class_id": int(detection['class']),
                "class": detection['name'],  # This will now have the swapped name
                "confidence": float(detection['confidence']),
                "box": [
                    float(detection['xmin']),
                    float(detection['ymin']),
                    float(detection['xmax']),
                    float(detection['ymax'])
                ]
            })
        
        # Render with labels (now showing correct swapped labels)
        results.render()  # Labels will be correct now
        annotated_results = results.ims[0]
        
        # Encode image as base64 to send via JSON
        success, buffer = cv2.imencode('.jpg', annotated_results)
        if not success:
            raise ValueError("Could not encode image")
            
        encoded_img = base64.b64encode(buffer).decode('utf-8')

        return {
            "detections": detections, 
            "image": encoded_img,
            "total_detections": len(detections),
            "status": "success"
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise ValueError(f"Prediction failed: {str(e)}")