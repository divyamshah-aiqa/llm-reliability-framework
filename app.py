import gradio as gr
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

# ===== Recreate Model Architecture =====

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(384, 256),   # model.0
            nn.ReLU(),             # model.1
            nn.Dropout(0.1),       # model.2
            nn.Linear(256, 128),   # model.3
            nn.ReLU(),             # model.4
            nn.Linear(128, 4)      # model.5
        )

    def forward(self, x):
        return self.model(x)




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

        # --- Temperature scaling (soft calibration) ---
        temperature = 1.5
        logits = logits / temperature

        probs = torch.softmax(logits, dim=1)

        confidence, pred = torch.max(probs, 1)

        sorted_probs, _ = torch.sort(probs, descending=True)
        margin = sorted_probs[0][0] - sorted_probs[0][1]

    decision = label_map[pred.item()]
    confidence = float(confidence)
    margin = float(margin)

    # ==========================
    # Smart Calibration Logic
    # ==========================

        HIGH_CONF = 0.70
    MEDIUM_CONF = 0.50
    SAFE_MARGIN = 0.15

    # Strong signal → trust model
    if confidence >= HIGH_CONF and margin >= SAFE_MARGIN:
        final_decision = decision

    # Medium confidence but strong margin → allow respond
    elif confidence >= MEDIUM_CONF and margin >= SAFE_MARGIN:
        final_decision = "respond"

    # If model predicts defer with low certainty → clarify instead
    elif decision == "defer" and confidence < HIGH_CONF:
        final_decision = "ask_clarify"

    else:
        final_decision = decision


    # Risk tagging
    if final_decision == "respond":
        category = "trusted"
    elif final_decision == "ask_clarify":
        category = "ambiguous"
    elif final_decision == "defer":
        category = "high_risk"
    else:
        category = "noise"

    return final_decision, confidence, margin, category




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
