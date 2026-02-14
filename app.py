import gradio as gr
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ===== Recreate Model Architecture =====

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(384, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# ===== Load Embedding Model =====

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ===== Load Trained Weights =====

model = SimpleModel()
model.load_state_dict(torch.load("decision_model.pt", map_location=torch.device("cpu")))
model.eval()

label_map = {
    0: "respond",
    1: "ask_clarify",
    2: "defer",
    3: "silent"
}

# ===== Evaluation Function =====

def evaluate(text):

    if text.strip() == "":
        return "silent", 1.0, 1.0, "noise"

    emb = embedder.encode([text], convert_to_tensor=True)

    with torch.no_grad():
        logits = model(emb)
        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, 1)

        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = sorted_probs[0][0] - sorted_probs[0][1]

    decision = label_map[pred.item()]
    confidence = float(confidence)
    margin = float(margin)

    if decision == "respond":
        category = "trusted"
    elif decision == "ask_clarify":
        category = "ambiguous"
    elif decision == "defer":
        category = "high_risk"
    else:
        category = "noise"

    return decision, confidence, margin, category


demo = gr.Interface(
    fn=evaluate,
    inputs=gr.Textbox(label="User Input"),
    outputs=[
        gr.Textbox(label="Decision"),
        gr.Number(label="Confidence"),
        gr.Number(label="Decision Margin"),
        gr.Textbox(label="Risk Category")
    ],
    title="LLM Reliability Evaluation Framework",
    description="Confidence-calibrated decision gating system that evaluates whether an AI should respond, clarify, defer, or remain silent before LLM invocation."
)

demo.launch()
