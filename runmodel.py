import pickle
import torch
import torch.nn as nn

## --- model

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(8713, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load full model
model = model = torch.load("full_model.pth", map_location="cpu", weights_only=False)
model.eval()

# Prepare input
text = ["you have won a free prize"]
X = vectorizer.transform(text)
X = torch.tensor(X.toarray(), dtype=torch.float32)

# Predict
with torch.inference_mode():
    probs = torch.sigmoid(model(X))
    preds = (probs >= 0.5).long()

print("Spam" if preds.item() == 1 else "Not Spam")

