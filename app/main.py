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

def extract_features(samples, rate: int) -> dict[str, float]:
    import librosa
    return {
        'rms': float(librosa.feature.rms(y=samples).mean()),
        'spectral_centroid': float(librosa.feature.spectral_centroid(y=samples, sr=rate).mean()),
        'zero_crossing_rate': float(librosa.feature.zero_crossing_rate(samples).mean()),
    }

@app.post('/analyze')
async def analyze(audio: UploadFile = File(...)):
    import librosa
    with tempfile.NamedTemporaryFile(suffix=Path(audio.filename or '.wav').suffix, delete=False) as handle:
        handle.write(await audio.read()); path = handle.name
    try:
        samples, rate = librosa.load(path, sr=16000, mono=True)
        features = extract_features(samples, rate)
        return {'emotion': classify_features(features['rms'], features['spectral_centroid']), 'features': features}
    finally: Path(path).unlink(missing_ok=True)
