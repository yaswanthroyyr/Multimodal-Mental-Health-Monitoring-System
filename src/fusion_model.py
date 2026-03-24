import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiModalFusionNetwork(nn.Module):
    def __init__(self, num_classes=5, embed_dim=512, num_heads=4, layers=2):
        """
        Args:
            num_classes: Number of emotion categories (e.g., Neutral, Happy, Sad, Angry, Anxious)
            embed_dim: The common dimension we project all modalities into.
            num_heads: Number of attention heads (multi-perspective analysis).
            layers: Number of Transformer encoder layers.
        """
        super(MultiModalFusionNetwork, self).__init__()
        
        # --- 1. Dimension Projection Layers ---
        # Align all inputs to the same size (embed_dim = 512)
        self.text_proj = nn.Linear(768, embed_dim)   # RoBERTa
        self.audio_proj = nn.Linear(1024, embed_dim) # HuBERT
        self.video_proj = nn.Linear(768, embed_dim)  # ViT
        
        # Dropouts for regularization (prevents overfitting)
        self.dropout = nn.Dropout(0.1)
        
        # --- 2. The Cross-Modal Transformer ---
        # This layer allows Text to "talk" to Audio and Video
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        
        # --- 3. Classification Head ---
        # Takes the fused representation and outputs probability
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, 256), # Input is 3 tokens (Text, Audio, Video) flattened
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, text_emb, audio_emb, video_emb):
        """
        Forward pass of the fusion network.
        
        Inputs:
            text_emb:  (Batch, 768)
            audio_emb: (Batch, 1024) - assuming mean-pooled for this architecture
            video_emb: (Batch, 768)
        """
        
        # 1. Project features to common dimension (Batch, 512)
        t = self.text_proj(text_emb)
        a = self.audio_proj(audio_emb)
        v = self.video_proj(video_emb)
        
        # 2. Add modality-specific positional embeddings (optional, but good practice)
        # (Skipping for brevity, but helps model know which is which)
        
        # 3. Stack them into a sequence: Shape becomes (Batch, 3, 512)
        # The sequence is [Text_Token, Audio_Token, Video_Token]
        multimodal_sequence = torch.stack([t, a, v], dim=1)
        
        # 4. Apply Transformer (Self-Attention across modalities)
        # The model now learns relationships between modalities
        fused_features = self.transformer_encoder(multimodal_sequence)
        
        # 5. Flatten the output for classification
        # Shape: (Batch, 3*512) = (Batch, 1536)
        flattened = fused_features.view(fused_features.size(0), -1)
        
        # 6. Predict
        logits = self.classifier(flattened)
        
        return logits

# --- Sanity Check / Unit Test ---
if __name__ == "__main__":
    # Create the model
    model = MultiModalFusionNetwork(num_classes=5)
    print("✅ Fusion Model Initialized")
    
    # Create dummy data (Batch Size = 2)
    dummy_text = torch.randn(2, 768)
    dummy_audio = torch.randn(2, 1024)
    dummy_video = torch.randn(2, 768)
    
    # Forward pass
    output = model(dummy_text, dummy_audio, dummy_video)
    
    print(f"Output Logits Shape: {output.shape} (Should be [2, 5])")
    
    # Simulate a prediction
    probs = F.softmax(output, dim=1)
    print(f"Predicted Probabilities:\n{probs}")