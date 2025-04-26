import cv2
import tempfile
import os
from io import BytesIO
import base64
import time
from utils.detection import detect_objects
from PIL import Image
import numpy as np

def stream_video(uploaded_file, model, confidence_threshold=0.25):
    """
    Generator function to stream video frames with real-time detection
    
    Args:
        uploaded_file: The uploaded video file
        model: The loaded YOLO model
        confidence_threshold: Minimum confidence threshold for detection
        
    Yields:
        base64 encoded frames with detection results
    """
    # Save the uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        video_path = temp_file.name
    
    try:
        # Open the video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            yield "Error: Could not open video file."
            return
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to PIL image
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # Convert PIL image to bytes
            img_byte_arr = BytesIO()
            pil_img.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()
            
            # Create a mock file-like object for detection
            mock_file = BytesIO(img_byte_arr)
            mock_file.name = "frame.jpg"
            
            # Run detection on the frame
            results = detect_objects(model, mock_file)
            
            # Draw bounding boxes
            for detection in results:
                if detection["confidence"] >= confidence_threshold:
                    bbox = detection["bbox"]
                    class_name = detection["class"]
                    confidence = detection["confidence"]
                    
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(frame, f"{class_name} {confidence:.2f}", 
                               (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Convert the frame to base64 string for streaming
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Yield the frame as a data URL
            yield f"data:image/jpeg;base64,{frame_base64}"
            
            # Control the frame rate (adjust as needed)
            time.sleep(0.03)  # ~30 fps
            
        # Release resources
        cap.release()
        
    finally:
        # Clean up the temporary file
        if os.path.exists(video_path):
            os.remove(video_path)