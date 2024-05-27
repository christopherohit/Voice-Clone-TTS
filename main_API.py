from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware

from api.voiceClone import generate_2

from api.model import InputData
import sys

app = FastAPI()


@app.get("/")
async def read_root():
    message = f"Hello world! From FastAPI running on Uvicorn with Gunicorn"
    return {"message": message}

@app.post("/CloneAudio",
            response_class=FileResponse)
async def Clone_audio(text: str = Form(...),  file: UploadFile = File(...)):
    return await generate_2(text, file)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    # with open(output_path, 'rb') as f:
    #     output_audio = f.read()
        
    # return FileResponse(output_audio, filename='result.wav')