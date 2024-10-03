import torch
import json
import asyncio
from pydantic import BaseModel
from fastapi import FastAPI, Query, Request, Body
from routers import websocket
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from routers import langchain

from typing import List

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


class scriptReq(BaseModel):
    processedScript: List[str]

@app.post("/restructure_script")
async def restructure_script(script_Req : scriptReq):
    print("[main]-[restructure_script] API Call")

    restructure_result = await asyncio.create_task(langchain.restructure_script(
        script_Req.processedScript
    ))
    
    return restructure_result


class problemReq(BaseModel):
    script: str
    subject : str
    problem_num : int
    problem_types : str

@app.post("/generateProblems")
async def generate_problems(problem_req: problemReq):
    print("[main]-[generate_problems] API Call")

    data = problem_req
    
    script = data.script
    subject = data.subject
    problem_num = data.problem_num
    problem_types = data.problem_types

    generate_result = await asyncio.create_task(langchain.generate_problems_full(
        script,
        subject,
        problem_num,
        problem_types
    ))
    
    return generate_result