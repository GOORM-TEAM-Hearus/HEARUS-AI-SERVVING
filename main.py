import torch
import asyncio
from fastapi import FastAPI, Query, Request
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
async def sttModification(text: str = Query(..., description="The text to be modified")):
    print("[main]-[sttModification] API Call :", text)

    llm_result = await asyncio.create_task(langchain.speech_to_text_modification(
        "connection_uuid", 
        text
    ))

    return llm_result


@app.post("/generateProblems")
async def generate_problems(request: Request):
    print(request)
    print(request.body)

    data = await request.json()

    print(data)
    
    script = data["script"]
    subject = data["subject"]
    problem_num = data["problem_num"]
    problem_types = data["problem_types"]

    generate_result = await asyncio.create_task(langchain.generate_problems(
        script,
        subject,
        problem_num,
        problem_types
    ))
    
    return generate_result