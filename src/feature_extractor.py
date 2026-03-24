import torch
import librosa
from PIL import Image
from transformers import (
    ViTImageProcessor, ViTForImageClassification,
    Wav2Vec2Processor, HubertModel,
    AutoTokenizer, AutoModelForSequenceClassification
)

class MultimodalFeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. VISUAL: Emotion-Specific Vision Transformer
        self.vit_processor = ViTImageProcessor.from_pretrained("dima806/facial_emotions_image_detection")
        self.vit_model = ViTForImageClassification.from_pretrained("dima806/facial_emotions_image_detection").to(self.device)
        
        # 2. AUDIO: HuBERT
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/hubert-large-ls960-ft")
        self.audio_model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft").to(self.device)

        # 3. TEXT: Emotion-Specific DistilRoBERTa
        self.text_tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.text_model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base").to(self.device)

    def process_video_frame(self, frame_path):
        """Returns emotion probabilities from the face."""
        image = Image.open(frame_path).convert("RGB")
        inputs = self.vit_processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vit_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs

    def process_text(self, text_input):
        """Returns emotion probabilities from text."""
        inputs = self.text_tokenizer(text_input, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.text_model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return probs

    def process_audio(self, audio_path):
        """Returns basic audio features."""
        speech, sr = librosa.load(audio_path, sr=16000)
        inputs = self.audio_processor(speech, sampling_rate=16000, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.audio_model(**inputs)
        return outputs.last_hidden_state