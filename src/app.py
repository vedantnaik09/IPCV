import streamlit as st
from utils.detection import load_model, detect_objects
from utils.websocket_handler import WebSocketHandler
import os
import tempfile
import cv2
import numpy as np
from PIL import Image
import io
import asyncio
import threading
import socket
import json
import sys
import time

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

# Create a session state for WebSocket server
if 'websocket_server_running' not in st.session_state:
    st.session_state.websocket_server_running = False
    st.session_state.websocket_thread = None
    st.session_state.websocket_handler = None

# Get local IP address for WebSocket connection
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't need to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# Function to run the WebSocket server
def run_websocket_server(model, confidence_threshold, port):
    # Configure asyncio to use a policy that works better with threads
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Create a new event loop for this thread
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    # Create WebSocket handler
    websocket_handler = WebSocketHandler(model, confidence_threshold)
    st.session_state.websocket_handler = websocket_handler
    
    # Run the server
    loop.run_until_complete(websocket_handler.start_server(port=port))
    
    # Keep the loop running
    try:
        loop.run_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        # Cleanup
        websocket_handler.stop()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        st.session_state.websocket_server_running = False

# Sidebar with options
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)

# WebSocket server controls
st.sidebar.header("Realtime Processing")
websocket_port = st.sidebar.number_input("WebSocket Port", min_value=1024, max_value=65535, value=8765)

# Start/Stop WebSocket server
if not st.session_state.websocket_server_running:
    start_button = st.sidebar.button("Start WebSocket Server")
    
    if start_button:
        # Create a status text area to display information
        status_text = st.sidebar.empty()
        status_text.info("Starting WebSocket server...")
        
        # Start the WebSocket server in a separate thread
        websocket_thread = threading.Thread(
            target=run_websocket_server, 
            args=(model, confidence_threshold, websocket_port)
        )
        websocket_thread.daemon = True
        websocket_thread.start()
        st.session_state.websocket_thread = websocket_thread
        
        # Wait a moment for the server to start
        time.sleep(2)
        
        # Update the session state
        st.session_state.websocket_server_running = True
        status_text.success("WebSocket server started!")
        
        # Rerun the app to reflect the new state
        st.rerun()
else:
    stop_button = st.sidebar.button("Stop WebSocket Server")
    
    if stop_button:
        # Create a status text area
        status_text = st.sidebar.empty()
        status_text.info("Stopping WebSocket server...")
        
        # Stop the WebSocket server
        if st.session_state.websocket_handler:
            st.session_state.websocket_handler.stop()
        
        # Update the session state
        st.session_state.websocket_server_running = False
        
        # Rerun the app to reflect the new state
        time.sleep(1)
        st.rerun()

# Display WebSocket connection info when server is running
if st.session_state.websocket_server_running:
    local_ip = get_local_ip()
    st.sidebar.success(f"WebSocket server running!")
    st.sidebar.info(f"Connect using: ws://{local_ip}:{websocket_port}")
    
    # Allow updating the confidence threshold for the WebSocket server
    if st.sidebar.button("Update WebSocket Confidence Threshold"):
        if st.session_state.websocket_handler:
            st.session_state.websocket_handler.confidence_threshold = confidence_threshold
            st.sidebar.success("Confidence threshold updated!")

# Create tabs for different functionalities
tab1, tab2, tab3 = st.tabs(["Upload Files", "Realtime Detection", "Help"])

with tab1:
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

with tab2:
    st.header("Realtime Helmet Detection")
    
    # Add HTML/JS for WebSocket client
    if st.session_state.websocket_server_running:
        local_ip = get_local_ip()
        websocket_url = f"ws://{local_ip}:{websocket_port}"
        
        st.markdown("""
        #### Connect to Webcam
        Click the button below to start streaming from your webcam. Detection results will be shown in real-time.
        """)
        
        # HTML and JavaScript for webcam streaming and WebSocket connection
        html_code = f"""
        <div>
            <div style="display: flex; flex-direction: row; align-items: flex-start;">
                <div style="flex: 1; margin-right: 10px;">
                    <h4>Webcam Feed</h4>
                    <video id="webcam" width="100%" autoplay></video>
                    <canvas id="canvas" style="display: none;"></canvas>
                    <button id="startBtn" style="margin-top: 10px;">Start Webcam</button>
                    <button id="stopBtn" style="margin-top: 10px; margin-left: 10px;" disabled>Stop Webcam</button>
                </div>
                <div style="flex: 1; margin-left: 10px;">
                    <h4>Detection Results</h4>
                    <img id="processedImage" width="100%" src="">
                    <div id="detections" style="margin-top: 10px; font-family: monospace;"></div>
                </div>
            </div>
        </div>

        <script>
            const websocketUrl = "{websocket_url}";
            let ws = null;
            let webcamStream = null;
            let isConnected = false;
            let isStreaming = false;

            // DOM elements
            const webcamEl = document.getElementById('webcam');
            const canvasEl = document.getElementById('canvas');
            const startBtn = document.getElementById('startBtn');
            const stopBtn = document.getElementById('stopBtn');
            const processedImageEl = document.getElementById('processedImage');
            const detectionsEl = document.getElementById('detections');

            // Initialize WebSocket connection
            function connectWebSocket() {{
                ws = new WebSocket(websocketUrl);
                
                ws.onopen = function() {{
                    console.log('WebSocket connected');
                    isConnected = true;
                }};
                
                ws.onmessage = function(event) {{
                    const data = JSON.parse(event.data);
                    if (data.status === 'success') {{
                        // Update the processed image
                        processedImageEl.src = data.processed_image;
                        
                        // Update detections list
                        let detectionsHtml = '';
                        if (data.detections && data.detections.length > 0) {{
                            detectionsHtml = '<ul>';
                            data.detections.forEach((detection, index) => {{
                                detectionsHtml += `<li>Detection #${{index+1}}: ${{detection.class}} (Confidence: ${{detection.confidence.toFixed(2)}})</li>`;
                            }});
                            detectionsHtml += '</ul>';
                        }} else {{
                            detectionsHtml = '<p>No objects detected.</p>';
                        }}
                        detectionsEl.innerHTML = detectionsHtml;
                    }} else if (data.status === 'error') {{
                        console.error('Error:', data.message);
                    }}
                }};
                
                ws.onclose = function() {{
                    console.log('WebSocket disconnected');
                    isConnected = false;
                    if (isStreaming) {{
                        stopWebcam();
                    }}
                }};
                
                ws.onerror = function(error) {{
                    console.error('WebSocket error:', error);
                }};
            }}

            // Start webcam stream
            async function startWebcam() {{
                try {{
                    webcamStream = await navigator.mediaDevices.getUserMedia({{ 
                        video: true 
                    }});
                    webcamEl.srcObject = webcamStream;
                    isStreaming = true;
                    startBtn.disabled = true;
                    stopBtn.disabled = false;
                    
                    // Connect to WebSocket if not already connected
                    if (!isConnected) {{
                        connectWebSocket();
                    }}
                    
                    // Start sending frames
                    sendFrames();
                }} catch (error) {{
                    console.error('Error accessing webcam:', error);
                    alert('Error accessing webcam: ' + error.message);
                }}
            }}

            // Stop webcam stream
            function stopWebcam() {{
                if (webcamStream) {{
                    webcamStream.getTracks().forEach(track => track.stop());
                    webcamEl.srcObject = null;
                    isStreaming = false;
                    startBtn.disabled = false;
                    stopBtn.disabled = true;
                    processedImageEl.src = '';
                    detectionsEl.innerHTML = '';
                }}
            }}

            // Send frames to the server
            function sendFrames() {{
                if (!isStreaming || !isConnected) return;
                
                const ctx = canvasEl.getContext('2d');
                canvasEl.width = webcamEl.videoWidth;
                canvasEl.height = webcamEl.videoHeight;
                ctx.drawImage(webcamEl, 0, 0, canvasEl.width, canvasEl.height);
                
                // Get frame as base64
                const frameData = canvasEl.toDataURL('image/jpeg', 0.8);
                
                // Send to server
                ws.send(JSON.stringify({{
                    type: 'frame',
                    data: frameData
                }}));
                
                // Schedule next frame
                setTimeout(sendFrames, 100); // Adjust for performance
            }}

            // Event listeners
            startBtn.addEventListener('click', startWebcam);
            stopBtn.addEventListener('click', stopWebcam);

            // Initial connection
            connectWebSocket();
        </script>
        """
        
        st.components.v1.html(html_code, height=700)
        
        # Add instructions for mobile streaming
        st.markdown("""
        ### Stream from Mobile Device
        
        To stream from a mobile device or another computer:
        
        1. Open a WebSocket client app on your device (like 'Simple WebSocket Client' browser extension)
        2. Connect to the WebSocket URL shown in the sidebar
        3. Take pictures or record video and send frames in this format:
        ```json
        {
          "type": "frame",
          "data": "data:image/jpeg;base64,<base64_encoded_image>"
        }
        ```
        4. You'll receive detection results back in JSON format
        """)

with tab3:
    st.header("Help & Instructions")
    
    st.markdown("""
    ### How to Use This App
    
    This app provides helmet detection using YOLO model in three ways:
    
    #### 1. Upload Files Tab
    - Upload images or videos for helmet detection
    - Adjust confidence threshold using the slider in the sidebar
    - View detection results with bounding boxes
    
    #### 2. Realtime Detection Tab
    - Start the WebSocket server from the sidebar
    - Use your webcam for realtime detection
    - Stream from mobile devices or other computers
    
    #### 3. WebSocket API
    For developers, you can connect to the WebSocket server programmatically:
    ```python
    import websockets
    import asyncio
    import json
    import base64
    
    async def send_image(image_path):
        with open(image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        async with websockets.connect("ws://YOUR_SERVER_IP:8765") as websocket:
            await websocket.send(json.dumps({
                "type": "frame",
                "data": f"data:image/jpeg;base64,{img_base64}"
            }))
            
            response = await websocket.recv()
            return json.loads(response)
    
    # Example usage
    asyncio.run(send_image("example.jpg"))
    ```
    """)

# Display example image if no file is uploaded
if 'uploaded_file' not in locals() or uploaded_file is None:
    # This is displayed in the Upload Files tab
    pass