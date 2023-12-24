from flask import Flask, request
from flask_cors import CORS
from models.stt_model import STTModel
from models.nlp_module import process_text

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
    processed_text = process_text(transcription)
    return processed_text


if __name__ == "__main__":
    app.run(port=5001)
