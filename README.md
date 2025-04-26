# Helmet Detection App

This project is a Streamlit application for detecting helmets in images and videos using a YOLOv8 model. Users can upload their media files, and the application will process them to identify whether helmets are present.

## Project Structure

```
helmet-detection-app
├── src
│   ├── app.py                # Main entry point for the Streamlit application
│   ├── utils
│   │   ├── __init__.py       # Initialization file for the utils package
│   │   └── detection.py       # Functions for loading the YOLO model and performing detection
│   └── assets
│       └── placeholder.jpg     # Placeholder image displayed when no image is uploaded
├── models
│   └── best.pt               # Saved YOLO model file for detection
├── uploads
│   └── .gitkeep              # Ensures the uploads directory is tracked by Git
├── requirements.txt           # Lists dependencies required for the project
└── README.md                  # Documentation for the project
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd helmet-detection-app
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```
   streamlit run src/app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`.

3. Use the interface to upload images or videos for helmet detection.

## Model

The application uses a YOLOv8 model trained to detect helmets. The model weights are stored in the `models/best.pt` file.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.