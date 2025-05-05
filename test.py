import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa

# Test whisper separately
def test_whisper():
    try:
        print("Loading Whisper tiny model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        print("Model loaded successfully")
        
        # Test with a simple audio file
        audio_path = "path/to/test.wav"  # Replace with a small audio file path
        print(f"Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path, sr=16000)
        print(f"Audio loaded, length: {len(audio)}, sample rate: {sr}")
        
        # Process with whisper
        input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
        print(f"Input features shape: {input_features.shape}")
        
        # Generate token ids
        predicted_ids = model.generate(input_features)
        print(f"Predicted IDs shape: {predicted_ids.shape}")
        
        # Decode
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        print(f"Transcription: {transcription}")
        
        return True
        
    except Exception as e:
        print(f"Error testing Whisper: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_whisper()