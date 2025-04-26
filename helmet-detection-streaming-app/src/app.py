import streamlit as st
from utils.detection import load_model, detect_objects
from utils.streaming import stream_video
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import io
import time
import requests
import subprocess
import threading

# Set page configuration
st.set_page_config(page_title="Helmet Detection Streaming App", layout="wide")

# Set the title of the Streamlit app
st.title("Helmet Detection Streaming App")

# Create uploads directory if it doesn't exist
# Use absolute path to avoid issues
uploads_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "uploads"))
os.makedirs(uploads_dir, exist_ok=True)

@st.cache_resource
def get_model():
    # Load the YOLO model
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best.pt"))
    return load_model(model_path)

model = get_model()

# Sidebar with options
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
mode = st.sidebar.radio("Mode", ["Image Upload", "Video Upload", "Real-time Streaming"])

if mode == "Image Upload":
    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        # Perform detection
        with st.spinner("Detecting objects..."):
            results = detect_objects(model, uploaded_file)
        
        # Display results on the image
        if results:
            # Read the image again
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Draw bounding boxes
            for detection in results:
                bbox = detection["bbox"]
                class_name = detection["class"]
                confidence = detection["confidence"]
                
                if confidence >= confidence_threshold:
                    cv2.rectangle(img_array, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                    cv2.putText(img_array, f"{class_name} {confidence:.2f}", 
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            with col2:
                st.subheader("Detection Results")
                st.image(img_array, caption='Detection Results', use_column_width=True)
            
            # Display detection information
            st.subheader("Detected Objects")
            for i, detection in enumerate(results):
                if detection["confidence"] >= confidence_threshold:
                    st.write(f"Detection #{i+1}: {detection['class']} (Confidence: {detection['confidence']:.2f})")
                    
        else:
            with col2:
                st.subheader("Detection Results")
                st.write("No objects detected.")

elif mode == "Video Upload":
    # File uploader for videos
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Create columns for display
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_file)
        
        # Process the video
        with st.spinner("Processing video... This may take a while."):
            # Save the video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
                temp_file.write(uploaded_file.getvalue())
                video_path = temp_file.name
            
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create a temporary file for the output video
            output_path = os.path.join(uploads_dir, f"output_{os.path.basename(video_path)}")
            
            # Define the codec and create VideoWriter object
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            frame_count = 0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Create a progress bar
            progress_bar = st.progress(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Convert frame to PIL image
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
                
                # Write the frame with detections
                out.write(frame)
                
                # Update progress
                frame_count += 1
                progress_bar.progress(frame_count / total_frames)
            
            # Release everything when done
            cap.release()
            out.release()
            
            # Display the processed video
            with col2:
                st.subheader("Detection Results")
                st.video(output_path)
            
            # Clean up temporary files
            os.remove(video_path)

elif mode == "Real-time Streaming":
    st.subheader("Real-time Video Streaming with Detection")
    
    # File uploader for videos to stream
    uploaded_file = st.file_uploader("Upload a video for streaming", type=["mp4", "avi", "mov"])
    
    # Add function to get local IP address
    import socket
    def get_local_ip():
        try:
            # Get the local IP address
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except:
            return "localhost"
    
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            video_path = temp_file.name
        
        # Get local IP for streaming
        server_ip = get_local_ip()
        stream_port = 5000  # Set a port for the streaming server
        stream_url = f"http://{server_ip}:{stream_port}/stream"
        
        st.info(f"Share this link to view the stream: {stream_url}")
        st.code(stream_url, language=None)
        
        # Import the streaming server
        from stream_server import start_stream_server
        
        stream_server_thread = None
        
        # Button to start streaming
        if st.button("Start Streaming"):
            # Create a placeholder for the local streaming view
            stream_placeholder = st.empty()
            
            # Start the streaming server in a separate thread
            if stream_server_thread is None or not stream_server_thread.is_alive():
                stream_server_thread = threading.Thread(
                    target=start_stream_server, 
                    args=(video_path, model, confidence_threshold, stream_port)
                )
                stream_server_thread.daemon = True
                stream_server_thread.start()
                
                st.success(f"Streaming server started! Access at {stream_url}")
                
                # Display an iframe to show the stream locally
                stream_placeholder.markdown(
                    f'<iframe src="{stream_url}" width="100%" height="480" allow="autoplay"></iframe>',
                    unsafe_allow_html=True
                )

else:
    # Display instructions when no mode is selected
    st.write("Please select a mode from the sidebar to get started.")
    
    # Display example image
    example_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "assets", "placeholder.jpg"))
    if os.path.exists(example_path):
        st.image(example_path, caption="Example image", use_column_width=True)