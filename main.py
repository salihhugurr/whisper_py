from flask import Flask, request, jsonify
import whisper
import os

app = Flask(__name__)
model = whisper.load_model("base")

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files or 'language' not in request.form:
        return jsonify({"error": "Audio file and language are required"}), 400
    
    audio_file = request.files['audio']
    language = request.form['language']

    # Ses dosyasını geçici bir dosyaya kaydet
    audio_path = os.path.join("/tmp", audio_file.filename)
    audio_file.save(audio_path)

    # Whisper modeli ile transkripsiyon yap
    result = model.transcribe(audio_path, language=language)

    # Geçici dosyayı sil
    os.remove(audio_path)

    return jsonify({"text": result['text']})

if __name__ == '__main__':
    app.run(debug=True)
