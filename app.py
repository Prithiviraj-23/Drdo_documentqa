import os
from flask import Flask, request, jsonify, render_template
from langflow.load import run_flow_from_json

app = Flask(__name__)

# Define the TWEAKS dictionary
TWEAKS = {
    "Prompt-2y08E": {
        "Document": "",
        "Question": "",
        "template": "Answer user's questions based on the document below:\n\n\nother than give \"I don't Know\"\n\n---\n\n{Document}\n\n---\n\nQuestion:\n{Question}\n\nAnswer:\n"
    },
    "ChatInput-FGACD": {
        "files": "",
        "input_value": "",
        "sender": "User",
        "sender_name": "User",
        "session_id": "",
        "should_store_message": True
    },
    "ChatOutput-q0cBN": {
        "data_template": "{text}",
        "input_value": "",
        "sender": "Machine",
        "sender_name": "AI",
        "session_id": "",
        "should_store_message": True
    },
    "ParseData-5v6pl": {
        "sep": "\n",
        "template": "{text}"
    },
    "File-ADzPj": {
        "path": r"C:\Users\prith\OneDrive\Desktop\chat_bot\J.M. Smith, Hendrick Van Ness, Michael Abbott, Mark Swihart - Introduction to Chemical Engineering Thermodynamics-McGraw-Hill Education (2018)1.txt",
        "silent_errors": False
    },
    "GoogleGenerativeAIModel-7CteZ": {
        "google_api_key":"AIzaSyB5hNTMw1Obh5tXn0zZAEawv0tmzWwtm24",  # Use environment variable for API key
        "input_value": "",
        "max_output_tokens": None,
        "model": "gemini-1.5-flash",
        "n": None,
        "stream": False,
        "system_message": "",
        "temperature": 0.1,
        "top_k": None,
        "top_p": None
    }
}

def run_langflow(message):
    TWEAKS["Prompt-2y08E"]["Question"] = message
    TWEAKS["ChatInput-FGACD"]["input_value"] = message
    
    result = run_flow_from_json(flow="Document QA (1).json",
                                input_value=message,
                                fallback_to_env_vars=True,
                                tweaks=TWEAKS)
    return result

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
        else:
            print("Response is empty or not in expected format.")
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

    result = main(question)
    
    return jsonify({'text': result})

def main(prompt):
    if prompt:
        response = run_langflow(prompt)
        text_output = extract_text(response)
        return text_output
    else:
        return "Please enter a question before submitting."

if __name__ == "__main__":
    # Use host='0.0.0.0' and port from environment variable
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
