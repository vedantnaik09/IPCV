from flask import Flask, Response
import cv2
from utils.detection import load_model, detect_objects
import io
from PIL import Image
import numpy as np
import threading
import os

app = Flask(__name__)
video_capture = None
model = None
confidence_threshold = 0.25

def gen_frames():
    global video_capture, model, confidence_threshold
    if video_capture is None:
        return
    
    while True:
        success, frame = video_capture.read()
        if not success:
            break
        
        # Convert frame to PIL image for detection
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # Convert PIL image to bytes
        img_byte_arr = io.BytesIO()
        pil_img.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()
        
        # Create a mock file-like object for detection
        mock_file = io.BytesIO(img_byte_arr)
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
        
        # Convert the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Send frame to client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/stream')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def start_stream_server(video_path, loaded_model, threshold=0.25, port=5000):
    global video_capture, model, confidence_threshold
    video_capture = cv2.VideoCapture(video_path)
    model = loaded_model
    confidence_threshold = threshold
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=port, threaded=True)