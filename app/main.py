from fastapi import FastAPI, UploadFile, File
import tempfile
from pathlib import Path

app = FastAPI(title="Speech Emotion Recognition")
LABELS = ["angry", "calm", "happy", "sad", "fearful"]

@app.get('/health')
def health(): return {'status': 'ok'}

def classify_features(rms: float, spectral_centroid: float) -> str:
    if rms > 0.12 and spectral_centroid > 2500: return 'angry'
    if rms < 0.03: return 'calm'
    if spectral_centroid > 2200: return 'happy'
    return 'sad'

@app.post('/analyze')
async def analyze(audio: UploadFile = File(...)):
    import librosa
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or '.wav').suffix, delete=False) as handle:
        handle.write(await audio.read()); path = handle.name
    try:
        samples, rate = librosa.load(path, sr=16000, mono=True)
        rms = float(librosa.feature.rms(y=samples).mean())
        centroid = float(librosa.feature.spectral_centroid(y=samples, sr=rate).mean())
        return {'emotion': classify_features(rms, centroid), 'features': {'rms': rms, 'spectral_centroid': centroid}}
    finally: Path(path).unlink(missing_ok=True)
