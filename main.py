from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import whisper
import os
import shutil

app = FastAPI()
model = whisper.load_model("base")

@app.post("/transcribe")
async def transcribe(audio: UploadFile, language: str = Form(...)):
    if not audio or not language:
        raise HTTPException(status_code=400, detail="Audio file and language are required")
    
    audio_path = f"/tmp/{audio.filename}"
    
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    result = model.transcribe(audio_path, language=language)

    os.remove(audio_path)

    return JSONResponse(content={"text": result['text']})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
