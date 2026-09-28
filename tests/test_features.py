from app.main import classify_features

def test_loud_high_frequency_voice_is_angry():
    assert classify_features(0.2, 3000) == 'angry'

def test_quiet_voice_is_calm():
    assert classify_features(0.01, 1000) == 'calm'
