# Helmet Detection Streaming App

This project is a real-time video streaming application that allows users to upload videos and view object detection results. It utilizes a pre-trained YOLO model for detecting objects in the uploaded video streams.

## Project Structure

```
helmet-detection-streaming-app
├── src
│   ├── app.py                # Main application file
│   ├── static
│   │   ├── css
│   │   │   └── style.css     # CSS styles for the application
│   │   └── js
│   │       └── main.js       # JavaScript for client-side functionality
│   ├── templates
│   │   ├── index.html        # Main HTML template for the application
│   │   └── stream.html       # HTML template for real-time video streaming
│   └── utils
│       ├── __init__.py       # Marks the utils directory as a Python package
│       ├── detection.py       # Functions for loading the YOLO model and detecting objects
│       └── streaming.py       # Functions for handling real-time video streaming
├── models
│   └── best.pt               # Pre-trained YOLO model for object detection
├── assets
│   └── placeholder.jpg        # Placeholder image for demonstration
├── uploads                    # Directory for temporarily storing uploaded video files
├── requirements.txt           # Python dependencies for the project
├── .gitignore                 # Files and directories to be ignored by Git
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd helmet-detection-streaming-app
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Ensure you have the necessary model file (`best.pt`) in the `models` directory.

## Usage

1. Run the application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Upload a video file to see real-time object detection results.

## Features

- Upload images and videos for object detection.
- Real-time video streaming with detection results.
- Adjustable confidence threshold for detections.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.