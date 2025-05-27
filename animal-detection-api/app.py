from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
import requests
from pathlib import Path
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
import os

env_path = Path(__file__).parent / ".env"
load_dotenv(dotenv_path=env_path, override=False)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_ENDPOINTS = {
    "herdnet": os.getenv("HERDNET_URL"),
    "maskrcnn": os.getenv("MASKRCNN_URL"),
    "detr": os.getenv("DETR_URL")
}

MAX_FILE_SIZE = 20 * 1024 * 1024
ALLOWED_CONTENT_TYPES = ["image/jpeg", "image/jpg"]


async def validate_upload_file(file: UploadFile) -> bytes:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Solo archivos JPEG son permitidos")

    total_size = 0
    chunks = []
    chunk_size = 1024 * 1024

    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        chunks.append(chunk)
        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="El archivo es demasiado grande, el tamaño máximo permitido es 20 MB")
        
    return b"".join(chunks)

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Form("herdnet")
):
    try:
        external_url = MODEL_ENDPOINTS.get(model)
        if not external_url:
            raise HTTPException(status_code=400, detail=f"Invalid model '{model}'")

        file_bytes = await validate_upload_file(file)

        files = {
            "file": (file.filename, file_bytes, file.content_type)
        }

        response = requests.post(external_url, files=files)
        response.raise_for_status()

        return JSONResponse(content=response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error forwarding request to {external_url}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Failed to forward request")
    except Exception as e:
        print(f"Internal server error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error")
