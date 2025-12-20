from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os
import logging

load_dotenv()

from chat import ChatService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

# Initialize chat service globally
chat_service = None

def init_chat_service():
    """Initialize the chat service with proper error handling."""
    global chat_service
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables")
        
        chat_service = ChatService(
            api_key=api_key,
            model="gemini-2.5-flash",
            max_memory=50,
            temperature=0.7
        )
        logger.info("Chat service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize chat service: {e}")
        chat_service = None


@app.before_request
def before_request():
    """Initialize chat service before first request."""
    global chat_service
    if chat_service is None:
        init_chat_service()


@app.route('/')
def index():
    """Serve the main chat interface."""
    try:
        return render_template('chat.html')
    except Exception as e:
        logger.error(f"Error rendering template: {e}")
        return jsonify({'error': 'Failed to load chat interface'}), 500


@app.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat messages and return responses."""
    try:
        # Validate request
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        payload = request.get_json(force=True)
        message = payload.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        if len(message) > 5000:
            return jsonify({'error': 'Message too long (max 5000 characters)'}), 400
        
        # Check if chat service is initialized
        if chat_service is None:
            logger.error("Chat service not initialized")
            return jsonify({'error': 'Chat service unavailable'}), 503
        
        # Generate response
        logger.info(f"Processing message: {message[:50]}...")
        reply = chat_service.chat(message)
        
        if not reply:
            return jsonify({'error': 'No response generated'}), 500
        
        logger.info(f"Response generated successfully")
        return jsonify({'reply': reply}), 200
    
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e)}), 400
    
    except RuntimeError as e:
        logger.error(f"Runtime error: {e}")
        return jsonify({'error': 'Failed to generate response'}), 500
    
    except Exception as e:
        logger.error(f"Unexpected error in chat endpoint: {e}")
        return jsonify({'error': 'An unexpected error occurred'}), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'chat_service_ready': chat_service is not None
    }), 200


@app.route('/clear', methods=['POST'])
def clear_conversation():
    """Clear conversation history."""
    try:
        if chat_service is None:
            return jsonify({'error': 'Chat service unavailable'}), 503
        
        chat_service.clear_memory()
        logger.info("Conversation history cleared")
        return jsonify({'message': 'Conversation cleared successfully'}), 200
    
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return jsonify({'error': 'Failed to clear conversation'}), 500


@app.route('/history', methods=['GET'])
def get_history():
    """Get conversation history."""
    try:
        if chat_service is None:
            return jsonify({'error': 'Chat service unavailable'}), 503
        
        history = chat_service.get_memory()
        return jsonify({'history': history}), 200
    
    except Exception as e:
        logger.error(f"Error retrieving history: {e}")
        return jsonify({'error': 'Failed to retrieve history'}), 500


@app.route('/export', methods=['GET'])
def export_conversation():
    """Export conversation as text."""
    try:
        if chat_service is None:
            return jsonify({'error': 'Chat service unavailable'}), 503
        
        export = chat_service.export_conversation()
        return jsonify({'export': export}), 200
    
    except Exception as e:
        logger.error(f"Error exporting conversation: {e}")
        return jsonify({'error': 'Failed to export conversation'}), 500


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    """Handle 405 errors."""
    return jsonify({'error': 'Method not allowed'}), 405


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_ENV', 'development') == 'development'
    
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)