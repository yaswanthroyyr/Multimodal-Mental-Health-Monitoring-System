import torch
import torch.nn as nn
import torch.optim as optim
from fusion_model import MultiModalFusionNetwork

# Configuration
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 5 # e.g. [Neutral, Happy, Sad, Angry, Anxious]

def train_dummy():
    # 1. Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiModalFusionNetwork(num_classes=NUM_CLASSES).to(device)
    
    # 2. Optimizer & Loss
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    
    print(f"🚀 Starting training on {device}...")
    model.train()
    
    # 3. Training Loop (Simulated)
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        
        # --- Simulate a Batch of Data ---
        # In reality, this comes from your DataLoader
        text_batch = torch.randn(BATCH_SIZE, 768).to(device)
        audio_batch = torch.randn(BATCH_SIZE, 1024).to(device)
        video_batch = torch.randn(BATCH_SIZE, 768).to(device)
        
        # Fake labels (0 to 4)
        labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(device)
        
        # --- Forward Pass ---
        outputs = model(text_batch, audio_batch, video_batch)
        
        # --- Loss Calculation ---
        loss = criterion(outputs, labels)
        
        # --- Backward Pass ---
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {loss.item():.4f}")

    print("✅ Training complete. Saving model...")
    torch.save(model.state_dict(), "models/fusion_model.pth")

if __name__ == "__main__":
    train_dummy()