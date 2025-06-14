# SayHi Servers

This repository contains the backend servers for the SayHi project, which provides real-time sign language detection and text-to-sign video conversion services.

## Components

- **realtime_detection.py**  
  WebSocket server for real-time sign language detection using MediaPipe and a trained Random Forest model.

- **text_to_sign_server.py**  
  Flask HTTP server that converts input text into a sign language video using pre-recorded video clips.

- **app.py**  
  Contains the core logic for text-to-sign conversion, including lemmatization and video concatenation.

- **process_manager.py**  
  Process manager to start, monitor, and restart the above servers for reliability.

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies (Flask, websockets, mediapipe, opencv-python, numpy, gradio, nltk, moviepy, etc.)

## Setup

1. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

2. **Prepare the dataset:**
   - Place sign language video clips in the `dataset/` directory.
   - Each video filename should correspond to a word or letter (e.g., `hello.mp4`, `a.mp4`).

3. **Model file:**
   - Place the trained model file `ASL_model.p` in the project root.

## Running the Servers

### Using the Process Manager

Start all servers and monitor them:
```
python process_manager.py
```

### Running Individually

- **Real-time Detection Server:**
  ```
  python realtime_detection.py
  ```
  - WebSocket endpoint: `ws://localhost:5001`

- **Text-to-Sign Server:**
  ```
  python text_to_sign_server.py
  ```
  - HTTP endpoint: `http://localhost:5000/convert`

## API Endpoints

### Real-time Detection (WebSocket)

- Connect to `ws://<host>:5001`
- Send frames as base64-encoded images in JSON.
- Receive predictions as JSON.

### Text-to-Sign (HTTP)

- `POST /convert`
  - JSON body: `{ "text": "your text here" }`
  - Returns: MP4 video file

- `GET /health`, `GET /test`
  - Health and test endpoints.

## Notes

- For best performance, run on a machine with a GPU.
- The process manager will automatically restart servers if they crash.
- For development, you can run `app.py` directly to use the Gradio interface.

## License

MIT License

