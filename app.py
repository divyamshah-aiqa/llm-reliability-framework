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
        probs = torch.softmax(logits, dim=1)

        confidence_tensor, pred = torch.max(probs, 1)

        sorted_probs, _ = torch.sort(probs, descending=True)
        margin_tensor = sorted_probs[0][0] - sorted_probs[0][1]

    predicted_class = label_map[pred.item()]
    confidence = float(confidence_tensor)
    margin = float(margin_tensor)

    # ----------------------------
    # Calibration Threshold Bands
    # ----------------------------

    HIGH_CONF = 0.70
    MEDIUM_CONF = 0.50

    SAFE_MARGIN = 0.25
    LOW_MARGIN = 0.10

    final_decision = predicted_class

    # ---- Zone 1: High Certainty ----
    if confidence >= HIGH_CONF and margin >= SAFE_MARGIN:
        final_decision = predicted_class

    # ---- Zone 2: Medium Certainty ----
    elif MEDIUM_CONF <= confidence < HIGH_CONF:
        if predicted_class == "respond":
            final_decision = "respond"
        else:
            final_decision = "ask_clarify"

    # ---- Zone 3: Low Certainty ----
    elif confidence < MEDIUM_CONF or margin < LOW_MARGIN:
        final_decision = "ask_clarify"

    # ----------------------------
    # Risk Category Tagging
    # ----------------------------

    if final_decision == "defer":
        category = "high_risk"
    elif confidence < MEDIUM_CONF:
        category = "low_confidence"
    elif margin < SAFE_MARGIN:
        category = "ambiguous"
    elif final_decision == "silent":
        category = "noise"
    else:
        category = "trusted"

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
