from typing import Union
from fastapi import FastAPI
from starlette.responses import FileResponse
from fastapi.staticfiles import StaticFiles 

app = FastAPI()

app.mount("/images", StaticFiles(directory="images"), name='images')

@app.get("/")
def read_root():
    return FileResponse('./templates/index.html')