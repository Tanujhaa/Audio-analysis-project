# AI-Driven Multi-Model Transformer Assessment for Teacher Performance Analysis System
import os
import torch
import numpy as np
import librosa
import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import pipeline
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tempfile
import base64
import time
import logging
import csv
import io
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set number of threads for CPU operations
torch.set_num_threads(4)

# Check if CUDA is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info(f"Using device: {device}")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size

# Initialize data directory and files
def init_data_directory():
    os.makedirs('data/images', exist_ok=True)
    csv_path = 'data/analysis_results.csv'
    if not os.path.exists(csv_path):
        with open(csv_path, 'w') as f:
            writer = csv.writer(f)
            writer.writerow([
                'result_id', 'teacher_id', 'audio_filename', 'upload_timestamp',
                'duration_seconds', 'speech_score', 'sentiment_score',
                'engagement_score', 'final_score', 'performance_level',
                'suggestions', 'transcription', 'visualization_filename'
            ])

init_data_directory()

# Load models with error handling
def load_models():
    try:
        logger.info("Loading Whisper model...")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny").to(device)
        logger.info("Whisper model loaded successfully!")
        return processor, model
    except Exception as e:
        logger.error(f"Failed to load Whisper model: {str(e)}")
        raise

try:
    whisper_processor, whisper_model = load_models()
    logger.info("Loading sentiment analysis model...")
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=0 if device == 'cuda' else -1
    )
    logger.info("Models loaded successfully!")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    raise

class AudioAnalysisModel(nn.Module):
    def __init__(self, mfcc_dim=40, text_dim=768, hidden_dim=256):
        super().__init__()
        self.mfcc_proj = nn.Linear(mfcc_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 2),
            nn.Sigmoid()
        )
        
    def forward(self, mfcc_features, text_features):
        mfcc = self.mfcc_proj(mfcc_features)
        text = self.text_proj(text_features)
        combined = (mfcc + text) / 2
        return self.output(combined) * 10  # Scale to 0-10

model = AudioAnalysisModel().to(device)

def get_next_result_id():
    csv_path = 'data/analysis_results.csv'
    if not os.path.exists(csv_path):
        return 1
    with open(csv_path) as f:
        return sum(1 for _ in csv.DictReader(f)) + 1

def save_results(data):
    csv_path = 'data/analysis_results.csv'
    fieldnames = [
        'result_id', 'teacher_id', 'audio_filename', 'upload_timestamp',
        'duration_seconds', 'speech_score', 'sentiment_score',
        'engagement_score', 'final_score', 'performance_level',
        'suggestions', 'transcription', 'visualization_filename'
    ]
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(data)

def validate_audio_file(audio_path):
    try:
        if not os.path.exists(audio_path):
            raise ValueError("File does not exist")
        if os.path.getsize(audio_path) == 0:
            raise ValueError("File is empty")
        
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(y) < 16000:
            raise ValueError("Audio too short (minimum 1 second required)")
        return True
    except Exception as e:
        logger.error(f"Audio validation failed: {str(e)}")
        raise

def process_audio(audio_path):
    try:
        validate_audio_file(audio_path)
        y, sr = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        mfcc_features = np.mean(mfccs, axis=1)
        mfcc_features = (mfcc_features - np.mean(mfcc_features)) / (np.std(mfcc_features) + 1e-8)
        
        input_features = whisper_processor(y, sampling_rate=16000, return_tensors="pt").input_features.to(device)
        predicted_ids = whisper_model.generate(input_features)
        transcription = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        chunks = [transcription[i:i+500] for i in range(0, len(transcription), 500)]
        results = [sentiment_analyzer(chunk)[0] for chunk in chunks if chunk.strip()]
        
        sentiment_score = 5.0
        if results:
            avg_score = sum(r['score'] for r in results) / len(results)
            sentiment_score = avg_score * 10 if results[0]['label'] == 'POSITIVE' else (1 - avg_score) * 10
        
        mfcc_tensor = torch.tensor(mfcc_features, dtype=torch.float32).unsqueeze(0).to(device)
        text_embedding = torch.randn(1, 768, device=device)
        
        with torch.no_grad():
            outputs = model(mfcc_tensor, text_embedding)
            speech_score = outputs[0, 0].item()
            engagement_score = outputs[0, 1].item()
        
        return {
            "transcription": transcription,
            "sentiment_score": sentiment_score,
            "speech_score": speech_score,
            "engagement_score": engagement_score
        }
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise

def generate_report(results):
    try:
        final_score = (results['speech_score'] + results['sentiment_score'] + results['engagement_score']) / 3
        
        suggestions = []
        if results['speech_score'] < 6:
            suggestions.append({"area": "Speech Clarity", "suggestion": "Practice clearer enunciation and pacing"})
        if results['sentiment_score'] < 6:
            suggestions.append({"area": "Emotional Expression", "suggestion": "Use more varied vocal tones"})
        if results['engagement_score'] < 6:
            suggestions.append({"area": "Engagement", "suggestion": "Add more interactive elements"})
        
        if final_score >= 8: level = "Excellent"
        elif final_score >= 7: level = "Very Good"
        elif final_score >= 6: level = "Good"
        elif final_score >= 5: level = "Average"
        else: level = "Needs Improvement"
        
        plt.figure(figsize=(10, 6))
        metrics = ['Speech', 'Sentiment', 'Engagement', 'Overall']
        scores = [results['speech_score'], results['sentiment_score'], results['engagement_score'], final_score]
        colors = ['#4285f4', '#34a853', '#fbbc05', '#ea4335']
        
        plt.bar(metrics, scores, color=colors)
        plt.ylim(0, 10)
        plt.title('Teaching Performance Metrics')
        plt.ylabel('Score (0-10)')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches='tight')
            plt.close()
            with open(tmp.name, 'rb') as f:
                viz = base64.b64encode(f.read()).decode('utf-8')
        os.unlink(tmp.name)
        
        return {
            **results,
            "final_score": round(final_score, 2),
            "performance_level": level,
            "suggestions": suggestions,
            "visualization": viz
        }
    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        raise

@app.route('/')
def home():
    return render_template('page.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    teacher_id = request.form.get('teacher_id', '1')  # Default to teacher_id=1
    
    try:
        filename = secure_filename(file.filename)
        ext = os.path.splitext(filename)[1].lower()
        
        if ext not in {'.wav', '.mp3', '.m4a'}:
            return jsonify({"error": "Unsupported file type"}), 400
        
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            file.save(tmp.name)
            start = time.time()
            results = process_audio(tmp.name)
            report = generate_report(results)
            
            # Generate unique IDs and filenames
            result_id = get_next_result_id()
            viz_filename = f"viz_{result_id}.png"
            viz_path = os.path.join('data', 'images', viz_filename)
            
            # Save visualization image
            with open(viz_path, 'wb') as f:
                f.write(base64.b64decode(report['visualization']))
            
            # Save all results to single CSV
            save_results({
                'result_id': result_id,
                'teacher_id': teacher_id,
                'audio_filename': filename,
                'upload_timestamp': datetime.now().isoformat(),
                'duration_seconds': librosa.get_duration(filename=tmp.name),
                'speech_score': report['speech_score'],
                'sentiment_score': report['sentiment_score'],
                'engagement_score': report['engagement_score'],
                'final_score': report['final_score'],
                'performance_level': report['performance_level'],
                'suggestions': json.dumps(report['suggestions']),  # Store as JSON string
                'transcription': report['transcription'],
                'visualization_filename': viz_filename
            })
            
            report["processing_time"] = round(time.time() - start, 2)
            return jsonify(report)
            
    except Exception as e:
        logger.error(f"Analysis error: {str(e)}", exc_info=True)
        return jsonify({"error": str(e)}), 500
    finally:
        if 'tmp' in locals():
            os.unlink(tmp.name)

@app.route('/export/csv')
def export_csv():
    try:
        # Read CSV data
        csv_path = 'data/analysis_results.csv'
        df = pd.read_csv(csv_path)
        
        # Convert suggestions from JSON string to readable format
        if 'suggestions' in df.columns:
            df['suggestions'] = df['suggestions'].apply(lambda x: '\n'.join([f"{item['area']}: {item['suggestion']}" 
                                                                             for item in json.loads(x)]))
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        
        return send_file(
            io.BytesIO(csv_buffer.getvalue().encode()),
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'teacher_analysis_{datetime.now().strftime("%Y%m%d")}.csv'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/export/json')
def export_json():
    try:
        # Read CSV data
        csv_path = 'data/analysis_results.csv'
        df = pd.read_csv(csv_path)
        
        # Convert suggestions from JSON string back to list of dicts
        if 'suggestions' in df.columns:
            df['suggestions'] = df['suggestions'].apply(json.loads)
        
        data = df.to_dict(orient='records')
        
        return send_file(
            io.BytesIO(json.dumps(data, indent=2).encode()),
            mimetype='application/json',
            as_attachment=True,
            download_name=f'teacher_analysis_{datetime.now().strftime("%Y%m%d")}.json'
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)