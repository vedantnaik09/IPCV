import asyncio
import json
import logging
import websockets
import cv2
import numpy as np
import base64
from PIL import Image
import io
import uuid
import os
import threading

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebSocketHandler:
    def __init__(self, model, confidence_threshold=0.25):
        self.clients = {}
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.server = None
        self.is_running = False
        self.ready_event = threading.Event()  # Add an event for synchronization
    
    async def register(self, websocket, client_id=None):
        if client_id is None:
            client_id = str(uuid.uuid4())
        self.clients[client_id] = websocket
        logger.info(f"New client connected: {client_id}")
        return client_id
    
    async def unregister(self, client_id):
        if client_id in self.clients:
            del self.clients[client_id]
            logger.info(f"Client disconnected: {client_id}")
    
    async def process_frame(self, frame_data, client_id):
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(frame_data.split(',')[1])
            img_arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
            
            # Convert to RGB for detection
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Create a temporary file for YOLO processing
            temp_img_path = f"temp_{client_id}.jpg"
            cv2.imwrite(temp_img_path, img)
            
            try:
                # Run detection
                results = self.model.predict(source=temp_img_path, conf=self.confidence_threshold)
                
                # Process results
                detections = []
                for result in results:
                    boxes = result.boxes.cpu().numpy()
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].astype(int)
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = result.names[class_id]
                        
                        if confidence >= self.confidence_threshold:
                            # Draw on the image
                            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img, f"{class_name} {confidence:.2f}", 
                                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
                            detections.append({
                                "class": class_name,
                                "confidence": float(confidence),
                                "bbox": [int(x1), int(y1), int(x2), int(y2)]
                            })
            finally:
                # Ensure we clean up temp file regardless of detection results
                if os.path.exists(temp_img_path):
                    os.remove(temp_img_path)
            
            # Convert processed image back to base64
            _, buffer = cv2.imencode('.jpg', img)
            img_processed_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Return the processed image and detections
            return {
                "status": "success",
                "processed_image": f"data:image/jpeg;base64,{img_processed_b64}",
                "detections": detections
            }
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def handle_connection(self, websocket, path):
        client_id = await self.register(websocket)
        try:
            async for message in websocket:
                try:
                    # Parse the incoming message
                    data = json.loads(message)
                    
                    if data["type"] == "frame":
                        # Process video frame
                        result = await self.process_frame(data["data"], client_id)
                        await websocket.send(json.dumps(result))
                    elif data["type"] == "config":
                        # Update confidence threshold
                        if "confidence_threshold" in data:
                            self.confidence_threshold = float(data["confidence_threshold"])
                            await websocket.send(json.dumps({"status": "config_updated"}))
                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    await websocket.send(json.dumps({"status": "error", "message": "Invalid JSON"}))
                except Exception as e:
                    logger.error(f"Error processing message: {str(e)}")
                    await websocket.send(json.dumps({"status": "error", "message": str(e)}))
        finally:
            await self.unregister(client_id)
    
    async def start_server(self, host="0.0.0.0", port=8765):
        self.is_running = True
        try:
            # Create a new event loop for this thread
            self.server = await websockets.serve(self.handle_connection, host, port)
            logger.info(f"WebSocket server started at ws://{host}:{port}")
            
            # Signal that the server is ready
            self.ready_event.set()
            
            # Keep the server running
            while self.is_running:
                await asyncio.sleep(1)
                
            # Close server when done
            if self.server:
                self.server.close()
                await self.server.wait_closed()
                logger.info("WebSocket server stopped")
                
        except Exception as e:
            logger.error(f"WebSocket server error: {str(e)}")
            self.is_running = False
            
        return self.server
    
    def stop(self):
        self.is_running = False
        self.ready_event.clear()