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

# EventObject for Stopping Thread
stop_event = threading.Event()

def speechToText(whisper_model, websocket):
    print("[Whisper] STT Thread Executed")

    # The last time a recording was retrieved from the queue.
    phrase_time = None

     # Set Timeout, Transition List
    phrase_timeout = 3
    transcription = ['']

    while not stop_event.is_set():
        sleep(0.25)
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                phrase_complete = False
                # If enough time has passed between recordings, consider the phrase complete.
                # Clear the current working audio buffer to start over with the new data.
                if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                    phrase_complete = True
                # This is the last time we received new audio data from the queue.
                phrase_time = now
                
                # Combine audio data from queue
                audio_data = b''.join(data_queue.queue)
                data_queue.queue.clear()
                
                # Convert in-ram buffer to something the model can use directly without needing a temp file.
                # Convert data from 16 bit wide integers to floating point with a width of 32 bits.
                # Clamp the audio stream frequency to a PCM wavelength compatible default of 32768hz max.
                audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

                # Read the transcription.
                result = whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="ko")
                text = result['text'].strip()
                print("Text " + text)

                # If we detected a pause between recordings, add a new item to our transcription.
                # Otherwise edit the existing one.
                if phrase_complete:
                    transcription.append(text)
                    for line in transcription:
                        print(line)
                else: 
                    transcription[-1] = text
        except Exception as e:
            print(f"Error processing audio data: {e}")
            break
    

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
            print("Received Audio Blob " + str(data_queue.qsize()))
    except Exception as e:
        print(f"[WebSocket] WebSocket error: {e}")
    finally:
        print("[WebSocket] Connection Closed")
        stop_event.set()
        stt_thread.join()
        print("[Whisper] STT Thread Destroyed")