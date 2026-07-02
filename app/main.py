from __future__ import annotations

from fastapi import FastAPI, File, Form, UploadFile

from app.ocr_engine import DEFAULT_LANGS, recognize


app = FastAPI(title="Local GPU OCR", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ocr")
async def ocr(
    file: UploadFile = File(...),
    langs: str = Form(",".join(DEFAULT_LANGS)),
) -> dict:
    image_bytes = await file.read()
    lang_list = [item.strip() for item in langs.split(",") if item.strip()]
    return recognize(image_bytes, lang_list)
