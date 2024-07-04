from fastapi import APIRouter, WebSocket

import argparse
import os
import numpy as np
import torch
import whisper

import asyncio
from datetime import datetime, timedelta
from queue import Queue
from time import sleep
import threading

router = APIRouter()

# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()

# Thread safe Queue for passing result from STT Thread
result_queue = Queue()

# EventObject for Stopping Thread
stop_event = threading.Event()


def speechToText(whisper_model, stop_event):
    print("[Whisper] STT Thread Executed")

    while not stop_event.is_set():
        sleep(0.25)
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                # Combine audio data from queue
                # Make total size of audio_data multiple of 2
                total_size = sum(len(chunk) for chunk in data_queue.queue)
                if total_size % 2 != 0:
                    padding_size = 2 - (total_size % 2)
                    # Add padding bytes
                    audio_data = b"".join(data_queue.queue) + b"\0" * padding_size
                else:
                    audio_data = b"".join(data_queue.queue)

                data_queue.queue.clear()

                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = (
                    np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
                    / 32768.0
                )

                # Read the transcription.
                result = whisper_model.transcribe(
                    audio_np, fp16=torch.cuda.is_available(), language="ko"
                )
                transcrition_result = result["text"].strip()
                print("[Whisper] Transition Result '" + transcrition_result + "'")

                if transcrition_result != "":
                    result_queue.put(transcrition_result)

        except Exception as e:
            print(f"[Whisper] Error processing audio data: {e}")
            break

    print("[Whisper] STT Thread Destroyed")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("[WebSocket] Configuring BE Client WebSocket")
    await websocket.accept()

    # Load Model
    model = "medium"
    whisper_model = whisper.load_model(model)
    print("[Whisper] Model Loaded Successfully")

    # Accept WebSocket
    data = await websocket.receive_text()
    print("[WebSocket] BE Client [" + data + "] Accepted")

    # Execute STT Thread until WebSocket Disconnected
    stt_thread = threading.Thread(target=speechToText, args=(whisper_model, stop_event))
    stt_thread.start()

    # Receive AudioBlob
    try:
        while True:
            audioBlob = await websocket.receive_bytes()
            data_queue.put(audioBlob)

            while not result_queue.empty():
                print("[WebSocket] Send Result from Result_Queue")
                result = result_queue.get()
                await websocket.send_text(result)

            # Sleep for other async functions
            await asyncio.sleep(0)

    except Exception as e:
        print(f"[WebSocket] WebSocket error: {e}")
    finally:
        websocket.close()

        stop_event.set()
        stt_thread.join()

        # clear stop_event for next Socket Connection
        stop_event.clear()

        while not data_queue.empty():
            data_queue.get()

        while not result_queue.empty():
            result_queue.get()

        print("[WebSocket] Connection Closed")
