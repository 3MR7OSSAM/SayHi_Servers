import asyncio
import websockets
import cv2
import numpy as np
import mediapipe as mp
import pickle
import json
import base64
import time
import os
import sys
from collections import deque
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Print Python version and environment information
print(f"Python version: {sys.version}")
print(f"Running in directory: {os.getcwd()}")
print(f"Files in current directory: {os.listdir('.')}")
print(f"Environment variables: PORT={os.environ.get('PORT')}")

# Import OpenCV with fallbacks
try:
    print("Attempting to import OpenCV...")
    import cv2
    print(f"Successfully imported cv2, version: {cv2.__version__}")
except ImportError as e:
    print(f"Error importing OpenCV: {e}")
    sys.exit(1)

# Import other dependencies
try:
    import numpy as np
    import mediapipe as mp
    import pickle
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    sys.exit(1)

# Attempt to load the model
try:
    model_path = os.path.abspath("./ASL_model.p")
    print(f"Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Model file not found! Contents of directory: {os.listdir('.')}")
        sys.exit(1)
    
    print("Loading model...")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    rf_model = model["model"]
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Load the Random Forest model and labels
labels = {  
    "Alef": "Alef", "Beh": "Beh", "Teh": "Teh", "Theh": "Theh", "Jeem": "Jeem",
    "Hah": "Hah", "Khah": "Khah", "Dal": "Dal", "Thal": "Thal", "Reh": "Reh",
    "Zain": "Zain", "Seen": "Seen", "Sheen": "Sheen", "Sad": "Sad", "Dad": "Dad",
    "Tah": "Tah", "Zah": "Zah", "Ain": "Ain", "Ghain": "Ghain", "Feh": "Feh",
    "Qaf": "Qaf", "Kaf": "Kaf", "Lam": "Lam", "Meem": "Meem", "Noon": "Noon",
    "Heh": "Heh", "Waw": "Waw", "Yeh": "Yeh", "Al": "Al", "Laa": "Laa",
    "Teh_Marbuta": "Teh_Marbuta", "1": "Back Space", "2": "Clear", "3": "Space", "4": ""
}

letter_map = {  
    "Alef": "ا", "Beh": "ب", "Teh": "ت", "Theh": "ث", "Jeem": "ج", "Hah": "ح",
    "Khah": "خ", "Dal": "د", "Thal": "ذ", "Reh": "ر", "Zain": "ز", "Seen": "س",
    "Sheen": "ش", "Sad": "ص", "Dad": "ض", "Tah": "ط", "Zah": "ظ", "Ain": "ع",
    "Ghain": "غ", "Feh": "ف", "Qaf": "ق", "Kaf": "ك", "Lam": "ل", "Meem": "م",
    "Noon": "ن", "Heh": "ه", "Waw": "و", "Yeh": "ي", "Al": "ال", "Laa": "لا",
    "Teh_Marbuta": "ة"
}

# Initialize Mediapipe components
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# WebSocket server details
host = os.environ.get("HOST", "0.0.0.0")  # Listen on all interfaces by default
port = int(os.environ.get("PORT", 5001))
print(f"Server will listen on {host}:{port}")
print(f"WebSocket server will be accessible at:")
print(f"  - Local: ws://localhost:{port}")
print(f"  - Network: ws://192.168.163.103:{port}")

# Connection tracking
connected_clients = set()

class ClientSession:
    def __init__(self, websocket):
        self.websocket = websocket
        self.predicted_text = ""
        self.final_characters = ""
        self.last_prediction_time = time.time()
        self.prediction_history = deque(maxlen=5)  # Store last 5 predictions
        self.frame_queue = deque(maxlen=3)  # Limit frame queue size
        self.processing_lock = asyncio.Lock()
        self.performance_metrics = {
            'frames_processed': 0,
            'predictions_made': 0,
            'avg_processing_time': 0,
            'last_frame_time': time.time()
        }
        self.min_confidence = 0.6  # Dynamic confidence threshold
        self.prediction_interval = 1.5  # Dynamic prediction interval
        
        # Initialize MediaPipe hands model for this client
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.5
        )
        logger.info("MediaPipe hands model initialized for client")
        
    def cleanup(self):
        """Clean up resources when client disconnects"""
        if hasattr(self, 'hands') and self.hands:
            self.hands.close()
            self.hands = None
            logger.info("MediaPipe hands model closed for client")
        
    def update_performance_metrics(self, processing_time):
        self.performance_metrics['frames_processed'] += 1
        self.performance_metrics['avg_processing_time'] = (
            (self.performance_metrics['avg_processing_time'] * (self.performance_metrics['frames_processed'] - 1) + processing_time) 
            / self.performance_metrics['frames_processed']
        )
        
    def adapt_parameters(self):
        # Adapt confidence and interval based on performance
        avg_time = self.performance_metrics['avg_processing_time']
        if avg_time > 0.2:  # If processing is slow
            self.min_confidence = min(0.8, self.min_confidence + 0.05)
            self.prediction_interval = min(3.0, self.prediction_interval + 0.1)
        elif avg_time < 0.1:  # If processing is fast
            self.min_confidence = max(0.4, self.min_confidence - 0.05)
            self.prediction_interval = max(1.0, self.prediction_interval - 0.1)

# Helper function to decode base64 strings with error handling
def decode_frame(base64_data):
    try:
        # Remove data URL prefix if present
        if base64_data.startswith('data:image'):
            base64_data = base64_data.split(',')[1]
            
        frame_data = base64.b64decode(base64_data)
        np_arr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None

def get_prediction_stability(session, predicted_character):
    """Check if prediction is stable across recent frames"""
    session.prediction_history.append(predicted_character)
    if len(session.prediction_history) < 3:
        return False
    
    # Check if at least 60% of recent predictions are the same
    recent_predictions = list(session.prediction_history)[-3:]
    most_common = max(set(recent_predictions), key=recent_predictions.count)
    stability_ratio = recent_predictions.count(most_common) / len(recent_predictions)
    
    return stability_ratio >= 0.6 and most_common == predicted_character

async def process_frame_queue(session):
    """Process frames from queue asynchronously"""
    if not session.frame_queue or session.processing_lock.locked():
        return
        
    async with session.processing_lock:
        if not session.frame_queue:
            return
            
        # Process the most recent frame
        frame_data = session.frame_queue.pop()
        await process_single_frame(session, frame_data)

async def process_single_frame(session, frame_data):
    """Process a single frame"""
    start_time = time.time()
    
    try:
        frame = decode_frame(frame_data['data'])
        if frame is None:
            return

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_image = session.hands.process(frame_rgb)  # Use session's hands model
        hand_landmarks = processed_image.multi_hand_landmarks

        if hand_landmarks:
            for hand_landmark in hand_landmarks:
                x_coordinates = [landmark.x for landmark in hand_landmark.landmark]
                y_coordinates = [landmark.y for landmark in hand_landmark.landmark]
                min_x, min_y = min(x_coordinates), min(y_coordinates)

                normalized_landmarks = []
                for coordinates in hand_landmark.landmark:
                    normalized_landmarks.extend([
                        coordinates.x - min_x,
                        coordinates.y - min_y
                    ])

                sample = np.asarray(normalized_landmarks).reshape(1, -1)
                
                # Predict probabilities
                probabilities = rf_model.predict_proba(sample)[0]
                predicted_index = np.argmax(probabilities)
                accuracy = probabilities[predicted_index]
                
                if accuracy >= session.min_confidence:
                    predicted_character = rf_model.classes_[predicted_index]
                    
                    # Check timing and stability
                    current_time = time.time()
                    time_since_last = current_time - session.last_prediction_time
                    
                    if (time_since_last >= session.prediction_interval and 
                        get_prediction_stability(session, predicted_character)):
                        
                        if predicted_character != "4":
                            character_to_send = await handle_character_prediction(
                                session, predicted_character, accuracy
                            )
                            
                            if character_to_send is not None:
                                session.last_prediction_time = current_time
                                session.performance_metrics['predictions_made'] += 1
                                
                                # Send response
                                response = {
                                    "character": character_to_send,
                                    "confidence": round(accuracy, 2),
                                    "timestamp": int(current_time * 1000),
                                    "processing_time": round((time.time() - start_time) * 1000, 1)
                                }
                                
                                await session.websocket.send(json.dumps(response))
                                
                                # Brief delay to prevent overwhelming the client
                                await asyncio.sleep(0.1)

        # Update performance metrics
        processing_time = time.time() - start_time
        session.update_performance_metrics(processing_time)
        session.adapt_parameters()
        
    except Exception as e:
        logger.error(f"Error processing frame: {e}")

async def handle_character_prediction(session, predicted_character, accuracy):
    """Handle character prediction and return character to send"""
    character_to_send = None
    
    if predicted_character in "123":
        if predicted_character == "1":  # Back Space
            if session.final_characters:
                session.final_characters = session.final_characters[:-1]
                character_to_send = "Back Space"
        elif predicted_character == "2":  # Clear
            session.final_characters = ""
            character_to_send = "Clear"
        elif predicted_character == "3":  # Space
            session.final_characters += " "
            character_to_send = "Space"
    else:
        # Regular character
        mapped_char = letter_map.get(predicted_character, "")
        if mapped_char:
            session.final_characters += mapped_char
            character_to_send = mapped_char
    
    return character_to_send

async def handle_ping(websocket):
    """Handle ping messages for connection health"""
    try:
        pong_response = {
            "type": "pong",
            "timestamp": int(time.time() * 1000)
        }
        await websocket.send(json.dumps(pong_response))
    except Exception as e:
        logger.error(f"Error handling ping: {e}")

async def receive_frames(websocket):
    session = ClientSession(websocket)
    connected_clients.add(websocket)
    client_ip = websocket.remote_address[0] if websocket.remote_address else "unknown"
    
    try:
        logger.info(f"Client connected from {client_ip} - MediaPipe model started")
        
        async for message in websocket:
            try:
                json_data = json.loads(message)
                
                # Handle different message types
                if json_data.get('type') == 'ping':
                    await handle_ping(websocket)
                    continue
                
                # Handle frame data
                if 'data' in json_data:
                    # Add frame to queue (non-blocking)
                    session.frame_queue.append(json_data)
                    
                    # Process frame queue
                    asyncio.create_task(process_frame_queue(session))
                    
            except json.JSONDecodeError:
                logger.error("Received invalid JSON data")
                error_response = {
                    "error": "Invalid JSON format",
                    "timestamp": int(time.time() * 1000)
                }
                await websocket.send(json.dumps(error_response))
            except Exception as e:
                logger.error(f"Error processing message: {e}")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client {client_ip} disconnected normally")
    except Exception as e:
        logger.error(f"Client {client_ip} connection error: {e}")
    finally:
        # Clean up MediaPipe model before removing client
        session.cleanup()
        connected_clients.discard(websocket)
        logger.info(f"Client {client_ip} connection cleaned up - MediaPipe model stopped. Active connections: {len(connected_clients)}")

# Add periodic cleanup task
async def cleanup_task():
    while True:
        await asyncio.sleep(30)  # Run every 30 seconds
        # Log server statistics
        logger.info(f"Active connections: {len(connected_clients)}")

# Enhanced server with better error handling
async def main():
    # Start cleanup task
    asyncio.create_task(cleanup_task())
    
    async with websockets.serve(
        receive_frames, 
        host, 
        port,
        ping_interval=20,  # Send ping every 20 seconds
        ping_timeout=10,   # Wait 10 seconds for pong
        close_timeout=10,  # Close timeout
        max_size=10**7,    # 10MB max message size
        compression=None   # Disable compression for better performance
    ):
        logger.info(f"WebSocket server started on ws://{host}:{port}")
        logger.info(f"Server is accessible at:")
        logger.info(f"  - Local: ws://localhost:{port}")
        logger.info(f"  - Network: ws://192.168.163.103:{port}")
        logger.info("Server is ready to accept connections...")
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")