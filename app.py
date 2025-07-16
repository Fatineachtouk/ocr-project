import os
from process import llm
from dotenv import load_dotenv  
from flask import Flask, request, jsonify


load_dotenv()

app = Flask(__name__)

@app.route('/ocr_api', methods=['POST'])

def ocr_api():
    try:
        input_data = request.get_json()
        result = llm(input_data)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=int(os.environ.get("MAIN_API_PORT", 5000)))