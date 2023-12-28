from flask import Flask, request, jsonify
from flask_cors import CORS
from models.stt_model import STTModel
from models.nlp_module import process_text
import json


app = Flask(__name__)
CORS(app)


stt_model = STTModel(
    "./deepspeech/deepspeech-0.9.3-models.pbmm",
    "./deepspeech/deepspeech-0.9.3-models.scorer",
)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    audio_file = request.files["audio"]
    transcription = stt_model.transcribe(audio_file)
    app.logger.info("Transcription Result : " + transcription)
    return transcription


@app.route("/process", methods=["POST"])
def process():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    try:
        text_data = request.get_json()

        processed_text = process_text(text_data)
        return jsonify(processed_text)
    except json.JSONDecodeError as e:
        return jsonify({"error": "Invalid JSON"}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
