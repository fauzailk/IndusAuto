from flask import Flask, render_template, request, jsonify
import traceback

# --- Agent Core Import ---
# Make sure you have the 'agent_core.py' file in the same directory.
from agent_core import agentic_conversation_step


# --- Flask App Initialization ---
app = Flask(__name__)


# --- Routes ---
@app.route('/')
def index():
    """
    Serves the main HTML page of the chat interface.
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles the chat logic by receiving user messages and returning the agent's response.
    """
    try:
        # Get data from the incoming JSON request
        data = request.get_json()
        user_input = data.get("message", "")
        state = data.get("state", None)

        # Call your agent's main logic function
        result = agentic_conversation_step(user_input, state)

        # The result is expected to be a dictionary like:
        # {"reply": "...", "state": ..., "lang": "..."}
        return jsonify(result)

    except Exception as e:
        # Print the full error to the console for debugging
        print(traceback.format_exc())
        # Return a JSON error message to the frontend
        return jsonify({"reply": f"[SERVER ERROR] {str(e)}", "state": None, "lang": "en"}), 500


# --- Main Execution Block ---
if __name__ == '__main__':
    # When you run this script directly, the following code will execute.
    
    # Using host='0.0.0.0' makes the server accessible from other devices on your network.
    # Your computer's firewall might ask for permission.
    # You can access it via http://127.0.0.1:5000 or http://localhost:5000 on this computer,
    # or using your computer's local IP address from other devices.
    print(" * Starting Flask server...")
    print(" * Access it from your browser at http://127.0.0.1:5000")
    app.run(host='0.0.0.0', port=5000)
