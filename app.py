from flask import Flask, request, jsonify
from flask_cors import CORS
from models.stt_model import STTModel
from models.nlp_module import process_text
from models.hugging_face import add_comment
import json

import whisper
import numpy as np
from pydub import AudioSegment
import io


app = Flask(__name__)
CORS(app)


stt_model = STTModel(
    "./deepspeech/deepspeech-0.9.3-models.pbmm",
    "./deepspeech/deepspeech-0.9.3-models.scorer",
)

# Choose a model size ("tiny", "base", "small", "medium", "large")
whisper_model = whisper.load_model("base")


@app.route("/transcribe", methods=["POST"])
def transcribe():
    # Get the audio file from the request
    audio_file = request.files["audio"]

    # Convert the audio file to the appropriate format
    audio_format = audio_file.filename.split(".")[-1]
    audio_segment = AudioSegment.from_file(
        io.BytesIO(audio_file.read()), format=audio_format
    )

    # Convert to mono and the required sample rate (16kHz for Whisper)
    audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)

    # Convert to numpy array
    audio_numpy = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    audio_numpy = audio_numpy / np.iinfo(audio_segment.array_type).max  # Normalize

    # Transcribe the audio file using Whisper
    result = whisper_model.transcribe(audio_numpy, language="ko")
    transcription = result["text"]

    app.logger.info("Transcription Result: " + transcription)
    return transcription


@app.route("/process", methods=["POST"])
def process():
    if not request.is_json:
        return jsonify({"error": "Missing JSON in request"}), 400

    try:
        text_data = request.get_json()

        processed_text = process_text(text_data)
        processed_text = add_comment(processed_text)
        return jsonify(processed_text)
    except json.JSONDecodeError as e:
        return jsonify({"error": "Invalid JSON"}), 400


if __name__ == "__main__":
    app.run(port=5000, debug=True)
