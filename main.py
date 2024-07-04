import torch
from fastapi import FastAPI, Query
from routers import websocket
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import langchain

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    # Should be edited in production env
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(websocket.router)
app.mount("/images", StaticFiles(directory="images"), name="images")


@app.get("/")
def read_root():
    return FileResponse("./templates/index.html")


@app.get("/sttModification")
def sttModification(text: str = Query(..., description="The text to be modified")):
    print("[main]-[sttModification] API Call :", text)
    return langchain.speech_to_text_modification("example_connection_uuid", text)
