import os
import tempfile
from ultralytics import YOLO
import cv2
import numpy as np

def load_model(model_path):
    """Load the YOLO model from the given path."""
    return YOLO(model_path)

def detect_objects(model, uploaded_file):
    """Detect objects in the uploaded file."""
    # Create a temporary directory to save the uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        # Write the uploaded file to the temporary file
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        # Run detection on the saved file
        results = model.predict(source=tmp_file_path, conf=0.25)
        
        # Process and return results
        detections = []
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                detections.append({
                    "class": class_name,
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })
        return detections
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_file_path):
            os.unlink(tmp_file_path)