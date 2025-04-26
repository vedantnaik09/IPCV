import streamlit as st
from utils.detection import load_model, detect_objects
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import io

# Set page configuration
st.set_page_config(page_title="Helmet Detection App", layout="wide")

# Set the title of the Streamlit app
st.title("Helmet Detection App")

# Create uploads directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

@st.cache_resource
def get_model():
    # Load the YOLO model
    model_path = os.path.join("models", "best.pt")
    return load_model(model_path)

model = get_model()

# Sidebar with options
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# File uploader for images and videos
uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    # Create columns for display
    col1, col2 = st.columns(2)
    
    # Display the uploaded file
    file_type = uploaded_file.type
    
    if file_type.startswith('image'):
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
            
    elif file_type.startswith('video'):
        # Save the video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            video_path = temp_file.name
        
        with col1:
            st.subheader("Original Video")
            st.video(uploaded_file)
        
        # Process the video
        with st.spinner("Processing video... This may take a while."):
            # Open the video file
            cap = cv2.VideoCapture(video_path)
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create a temporary file for the output video
            output_path = os.path.join("uploads", f"output_{os.path.basename(video_path)}")
            
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
else:
    # Display instructions when no file is uploaded
    st.write("Please upload an image or video to perform helmet detection.")
    
    # Display example image
    example_path = os.path.join("assets", "placeholder.jpg")
    if os.path.exists(example_path):
        st.image(example_path, caption="Example image", use_column_width=True)