# Speech Emotion Recognition

Librosa feature pipeline that classifies vocal tone from audio. This MVP exposes interpretable RMS and spectral-centroid features; a trained classifier can replace the rule boundary without changing the API.

```mermaid
flowchart LR
  Audio --> Librosa[Waveform features]
  Librosa --> Classifier[Emotion classifier]
  Classifier --> API[FastAPI result]
```

Run `pip install -r requirements.txt`, `uvicorn app.main:app --reload`, then POST audio to `/analyze`.
