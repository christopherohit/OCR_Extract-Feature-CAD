from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
import uvicorn
from starlette.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
import sys
from src.cad_ocr.ocr import OCR
from api_core.ocr_api import process



app = FastAPI()

@app.get('/')
async def read_root():
    message = 'Hello World,  From FastAPI running on Uvicorn with Gunicorn'
    return {"message":message}

@app.post('/ocr-cad', response_class=FileResponse)
async def OCR_CAD(file: UploadFile = File(...)):
    return await process(file= file)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# if __name__ == '__main__':

#     print('Save PIL to cache file')
    
#     file_cache = main_process.save_pil_to_file(pil_image= pil_image)
#     print('Save PIL to cache file done')
#     print('Start Image resolution')
#     image_high_resolution = main_process.upscale_image(image_path= file_cache)
