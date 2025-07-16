import os
import traceback
from dotenv import load_dotenv
from french_htr import kraken_ocr
from flask import Flask, request, jsonify

load_dotenv()


app = Flask(__name__)

DEFAULT_MODEL_PATH = os.environ["KRAKEN_MODEL_PATH"]


@app.route('/ocr', methods=['POST'])
def ocr_endpoint():
    try:
        data = request.json

        if 'image_path' not in data:
            return jsonify({"error": "Missing 'image_path'"}), 400

        image_path = data['image_path']
        model_path = data.get('model_path', DEFAULT_MODEL_PATH)

        result_text = kraken_ocr(image_path, model_path)
        return jsonify({"text": result_text})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get("KRAKEN_API_PORT", 5001)))

