from flask import Flask, request, jsonify
from french_htr import kraken_ocr
import traceback

app = Flask(__name__)

DEFAULT_MODEL_PATH = "peraire2_ft_MMCFR.mlmodel"

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
    app.run(host='127.0.0.1', port=5001, debug=True, use_reloader=False)

