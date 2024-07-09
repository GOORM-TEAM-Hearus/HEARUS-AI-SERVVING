from fastapi import APIRouter, WebSocket
import uuid
import json

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

from . import langchain

router = APIRouter()

# Thread safe Queue for passing data from the threaded recording callback.
data_queue = Queue()

# Thread safe Queue for passing result from STT Thread
result_queue = Queue()

# Thread safe Queue for passing result from Process Thread to LLM Thread
llm_queue = Queue()

# EventObject for Stopping Thread
stop_event = threading.Event()


def speechToText(whisper_model, stop_event):
    print("[STTThread] STT Thread Executed")

    max_audio_duration = 5  # 최대 오디오 길이 (초)
    sample_rate = 16000  # 오디오 샘플 레이트 (Hz)
    max_audio_size = max_audio_duration * sample_rate * 2  # 최대 오디오 크기 (바이트)

    while not stop_event.is_set():
        sleep(3)
        try:
            now = datetime.utcnow()
            # Pull raw recorded audio from the queue.
            if not data_queue.empty():
                # Combine audio data from queue up to max_audio_size
                audio_data = bytearray()
                while not data_queue.empty() and len(audio_data) < max_audio_size:
                    chunk = data_queue.get()
                    audio_data.extend(chunk)

                # Make total size of audio_data multiple of 2
                total_size = len(audio_data)
                if total_size % 2 != 0:
                    padding_size = 2 - (total_size % 2)
                    # Add padding bytes
                    audio_data.extend(b"\0" * padding_size)

                # Convert audio_data to bytes
                audio_data = bytes(audio_data)

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
                print("[STTThread] Transition Result '" + transcrition_result + "'")

                if transcrition_result != "":
                    result_queue.put(transcrition_result)

        except Exception as e:
            print(f"[STTThread] Error processing audio data: {e}")
            break

    print("[STTThread] STT Thread Destroyed")


class Message:
    def __init__(self, text_id, transcrition_result):
        self.text_id = text_id
        self.transcrition_result = transcrition_result


async def llm_thread(websocket: WebSocket, connection_uuid):
    print("[LLMTask] LLM Task Initiated")
    while not stop_event.is_set():
        await asyncio.sleep(0.25)
        try:
            if not llm_queue.empty():
                message = llm_queue.get()
                transcrition_result = message.transcrition_result
                text_id = message.text_id

                # langchain.speech_to_text_modification 함수를 별도의 비동기 작업으로 실행
                llm_result = await asyncio.create_task(langchain.speech_to_text_modification(
                    connection_uuid, 
                    transcrition_result
                ))

                if not llm_result:
                    llm_result = transcrition_result

                message_data = {
                    "textId": text_id,
                    "transcritionResult": llm_result
                }
                
                message_json = json.dumps(message_data, ensure_ascii=False)

                await websocket.send_text(message_json)
        except Exception as e:
            print(f"[LLMTask] Error : {e}")
            break
    
    print("[LLMTask] Process Task Destroyed")


async def process_thread(websocket: WebSocket):
    print("[ProcessTask] Process Task Initiated")
    while not stop_event.is_set():
        await asyncio.sleep(0.25)
        try:
            if not result_queue.empty():
                transcrition_result = result_queue.get()
                text_id = str(uuid.uuid4())
                
                message = Message(text_id, transcrition_result)

                llm_queue.put(message)
                
                # message_data = {
                #     "textId": text_id,
                #     "transcritionResult": transcrition_result
                # }
                
                # message_json = json.dumps(message_data, ensure_ascii=False)

                # await websocket.send_text(message_json)
        except Exception as e:
            print(f"[ProcessTask] Error : {e}")
            break
    
    print("[ProcessTask] Process Task Destroyed")


async def websocket_task(websocket: WebSocket):
    # Load Model
    model = "medium"

    # enforece GPU if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    whisper_model = whisper.load_model(model, device=device)
    print("[WebSocketTask]-[Whisper] Model Loaded Successfully with", device)

    # Accept WebSocket
    connection_uuid = await websocket.receive_text()
    print("[WebSocketTask] Connection [" + connection_uuid + "] Accepted")

    # Execute STT Thread until WebSocket Disconnected
    sttThread = threading.Thread(target=speechToText, args=(whisper_model, stop_event))
    sttThread.start()

    llmTask = asyncio.create_task(llm_thread(websocket, connection_uuid))

    processTask = asyncio.create_task(process_thread(websocket))

    # Receive AudioBlob
    try:
        while True:
            audioBlob = await websocket.receive_bytes()
            data_queue.put(audioBlob)

            # Sleep for other async functions
            await asyncio.sleep(0)

    except Exception as e:
        print(f"[WebSocketTask] WebSocket error: {e}")
    finally:
        stop_event.set()
        sttThread.join()
        llmTask.cancel()
        processTask.cancel()

        # clear stop_event for next Socket Connection
        stop_event.clear()

        while not data_queue.empty():
            data_queue.get()

        while not result_queue.empty():
            result_queue.get()

        while not llm_queue.empty():
            llm_queue.get()
        
        langchain.delete_data_by_uuid(connection_uuid)

        if websocket.client_state.name != "DISCONNECTED":
            await websocket.close()

        print("[WebSocketTask] Connection Closed")


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print("[WebSocket] Configuring BE Client WebSocket")
    await websocket.accept()

    print("[WebSocket] Configuring WebSocket Task")
    await asyncio.create_task(websocket_task(websocket))