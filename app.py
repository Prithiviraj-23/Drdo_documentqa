import os
from flask import Flask, request, jsonify, render_template
from langflow.load import run_flow_from_json

app = Flask(__name__)
file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "J.M. Smith, Hendrick Van Ness, Michael Abbott, Mark Swihart - Introduction to Chemical Engineering Thermodynamics-McGraw-Hill Education (2018)1.txt")

# Define the TWEAKS dictionary
TWEAKS = {
    "Prompt-2y08E": {
        "Document": "",
        "Question": "",
        "template": "Answer user's questions based on the document below:\n\n\nother than give \"I don't Know\"\n\n---\n\n{Document}\n\n---\n\nQuestion:\n{Question}\n\nAnswer:\n"
    },
    "ChatInput-FGACD": {
        "input_value": "",  # Initialize with a default value
        "sender": "User",
        "sender_name": "User",
        "session_id": "",
        "should_store_message": True
    },
    "File-ADzPj": {
        "path": file_path,
        "silent_errors": False
    },
    "GoogleGenerativeAIModel-7CteZ": {
        "google_api_key":"AIzaSyB5hNTMw1Obh5tXn0zZAEawv0tmzWwtm24",  # Use environment variable for API key
        "temperature": 0.1
    }
}

def run_langflow(message):
    TWEAKS["Prompt-2y08E"]["Question"] = message
    TWEAKS["ChatInput-FGACD"]["input_value"] = message
    
    try:
        result = run_flow_from_json(flow="Document QA (1).json",
                                     input_value=message,
                                     fallback_to_env_vars=True,
                                     tweaks=TWEAKS)
        return result
    except Exception as e:
        print(f"Error in run_langflow: {e}")
        return None

def extract_text(response):
    try:
        if isinstance(response, list) and len(response) > 0:
            run_output = response[0]
            if hasattr(run_output, 'outputs'):
                outputs = run_output.outputs
                if outputs:
                    for output in outputs:
                        if hasattr(output, 'results'):
                            result_data = output.results
                            message_data = result_data.get("message", {})
                            if "text" in message_data.data:
                                return message_data.data["text"].strip()
    except Exception as e:
        print(f"Error parsing response: {e}")
    
    return "No valid response found."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        result = main(question)
        return jsonify({'text': result})
    except Exception as e:
        print(f"Error in ask_question: {e}")
        return jsonify({'error': 'Internal server error'}), 500

def main(prompt):
    if prompt:
        response = run_langflow(prompt)
        text_output = extract_text(response)
        return text_output
    else:
        return "Please enter a question before submitting."

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
