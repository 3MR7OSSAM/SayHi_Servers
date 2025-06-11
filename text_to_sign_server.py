from flask import Flask, request, send_file, after_this_request, jsonify
import os
from datetime import datetime
import uuid
from flask_cors import CORS

app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Access-Control-Allow-Origin", "Access-Control-Allow-Methods"],
        "expose_headers": ["Content-Length", "Content-Type"]
    }
})

# Import existing text-to-sign functions
from app import texttoSign

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'text-to-sign',
        'timestamp': datetime.utcnow().isoformat(),
        'version': '1.0.0'
    }), 200

@app.route('/', methods=['GET'])
def home():
    """Root endpoint with service information"""
    return jsonify({
        'service': 'Text to Sign Language Server',
        'status': 'running',
        'endpoints': {
            'convert': '/convert (POST)',
            'health': '/health (GET)',
            'test': '/test (GET)'
        },
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/test', methods=['GET'])
def test_endpoint():
    """Test endpoint for quick verification"""
    return jsonify({
        'message': 'Text to Sign Language Server is working!',
        'status': 'success',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/convert', methods=['POST'])
def convert_text():
    output_path = None
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        
        # Generate unique filename for concurrent requests
        unique_id = str(uuid.uuid4())
        output_path = f"output_{unique_id}.mp4"
        
        # Convert text to sign language video
        video_path = texttoSign(text, output_path)
        
        if video_path and os.path.exists(video_path):
            # Set up cleanup after request is complete
            final_path = video_path  # Store path for closure
            @after_this_request
            def cleanup(response):
                try:
                    if os.path.exists(final_path):
                        os.remove(final_path)
                        print(f"Cleaned up video file: {final_path}")
                except Exception as e:
                    print(f"Error cleaning up file: {e}")
                return response

            # Send the video file
            return send_file(video_path, mimetype='video/mp4')
        else:
            return jsonify({'error': 'Failed to generate video'}), 500

    except Exception as e:
        # Clean up if error occurs before sending response
        if output_path and os.path.exists(output_path):
            try:
                os.remove(output_path)
            except:
                pass
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Get configuration from environment
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    print(f"Starting Text to Sign Language Server...")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Debug: {debug}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)
