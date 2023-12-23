import deepspeech
import numpy as np
import wave


class STTModel:
    def __init__(self, model_path, scorer_path):
        self.model = deepspeech.Model(model_path)
        self.model.enableExternalScorer(scorer_path)

    def transcribe(self, audio_file):
        with wave.open(audio_file, "rb") as w:
            frames = w.getnframes()
            buffer = w.readframes(frames)
            data16 = np.frombuffer(buffer, dtype=np.int16)

        return self.model.stt(data16)
