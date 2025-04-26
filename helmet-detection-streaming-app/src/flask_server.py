from flask import Flask, Response, render_template
import cv2
import tempfile
import os
from io import BytesIO
import base64
import time
from utils.detection import load_model, detect_objects
from PIL import Image
import argparse
import threading

app = Flask(__name__)

# Global variables
video_path = None
model = None
confidence_threshold = 0.25
stream_active = False

@app.route('/')
def index():
    return render_template('stream_viewer.html')

def generate_frames():
    global video_path, model, confidence_threshold, stream_active
    
    if not video_path or not stream_active:
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        yield (b'--frame\r\n'
               b'Content-Type: text/plain\r\n\r\n'
               b'Error: Could not open video file.\r\n\r\n')
        return
    
    while cap.isOpened() and stream_active:
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
        
        # Convert the frame to jpg
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in the MIME multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        # Control the frame rate
        time.sleep(0.03)  # ~30 fps
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_stream/<path:video_path_arg>/<float:threshold>')
def start_stream(video_path_arg, threshold):
    global video_path, confidence_threshold, stream_active
    video_path = video_path_arg
    confidence_threshold = threshold
    stream_active = True
    return "Stream started"

@app.route('/stop_stream')
def stop_stream():
    global stream_active
    stream_active = False
    return "Stream stopped"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask Server for Video Streaming')
    parser.add_argument('--model', type=str, required=True, help='Path to the YOLO model')
    args = parser.parse_args()
    
    # Load model
    model = load_model(args.model)
    
    # Create a templates directory and add a stream_viewer.html file
    os.makedirs('templates', exist_ok=True)
    with open('templates/stream_viewer.html', 'w') as f:
        f.write('''
        <!DOCTYPE html>
        <html>
        <head>
            <title>Helmet Detection Stream</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    text-align: center;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                h1 { color: #333; }
                .video-container {
                    max-width: 800px;
                    margin: 20px auto;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }
                img {
                    width: 100%;
                    height: auto;
                }
            </style>
        </head>
        <body>
            <h1>Helmet Detection Live Stream</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" alt="Live Stream">
            </div>
        </body>
        </html>
        ''')
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)