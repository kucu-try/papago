from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from googletrans import Translator
import cv2
import numpy as np
from paddleocr import PaddleOCR
from pykospacing import Spacing
from transformers import T5Tokenizer, T5ForConditionalGeneration
import os
import uvicorn
from fastapi.staticfiles import StaticFiles

app = FastAPI()
# 정적 파일 서빙을 위한 설정
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
translator = Translator()

model_name = "j5ng/et5-typos-corrector"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
ocr = PaddleOCR(lang="korean")
spacing = Spacing()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # 'index.html'을 렌더링하고, 필요한 변수를 전달
    return templates.TemplateResponse("index.html", {"request": request, "translated_text": ""})

def correct_spacing(text):
    return spacing(text)

def correct_typo(text, model, tokenizer):
    inputs = tokenizer.encode("correct: " + text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=512, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def translate_text(text, dest_language='en'):
    # googletrans를 사용한 번역
    translated = translator.translate(text, dest=dest_language)
    return translated.text

def read_imagefile(file) -> np.ndarray:
    image = cv2.imdecode(np.fromstring(file, np.uint8), 1)
    return image

@app.post("/process-image/")
async def process_image(request: Request, file: UploadFile = File(...), target_lang: str = Form(...)):
    image_data = await file.read()
    image = read_imagefile(image_data)

    result = ocr.ocr(image, cls=False)
    extracted_text = " ".join([line[1][0] for line in result[0]])
    corrected_text = correct_spacing(extracted_text)
    final_text = correct_typo(corrected_text, model, tokenizer)
    translated_text = translate_text(final_text, dest_language=target_lang)

    return templates.TemplateResponse(
        "index.html",
        {"translated_text": translated_text, "request": request},
        media_type="text/html"
    )

@app.post("/process-text/")
async def process_text(request: Request, text_to_translate: str = Form(...), target_lang: str = Form(...)):
    # corrected_text = correct_spacing(text_to_translate)
    # final_text = correct_typo(corrected_text, model, tokenizer)
    translated_text = translate_text(text_to_translate, dest_language=target_lang)
    translated_lines = translated_text.splitlines()  # 번역된 텍스트를 줄별로 분할

    return templates.TemplateResponse(
        "index.html",
        {"translated_lines": translated_lines, "request": request},  # 템플릿에 줄별로 분할된 텍스트를 전달
        media_type="text/html"
    )

@app.post("/process-correct")
async def process_correct(request: Request, text_to_correct: str = Form(...)):
    corrected_text = correct_spacing(text_to_correct)
    final_text = correct_typo(corrected_text, model, tokenizer)

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "final_text": final_text},
        media_type="text/html"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
