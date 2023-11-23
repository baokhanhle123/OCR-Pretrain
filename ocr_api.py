# -*- coding: utf-8 -*-
from fastapi import FastAPI, UploadFile, File
from starlette.responses import FileResponse
from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np
from PIL import Image

app = FastAPI()

ocr = PaddleOCR(use_angle_cls=True, lang='en')
font_path = './fonts/simfang.ttf'

@app.post("/process_image")
async def process_image(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    result = ocr.ocr(img, cls=True)

    image = Image.open(img).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
    im_show = Image.fromarray(im_show)
    result_path = './result1.jpg'
    im_show.save(result_path)

    return {"filename": file.filename}

@app.get("/get_image/{filename}")
async def get_image(filename: str):
    return FileResponse(filename, media_type="image/jpeg")