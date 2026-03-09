# agentic_orchestration/agent_demo.py

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# --- Calibration guardrail logic (inline) ---
def margin_confidence(probs: torch.Tensor) -> float:
    """Return margin between top two probabilities."""
    sorted_probs, _ = torch.sort(probs, descending=True)
    return float(sorted_probs[0] - sorted_probs[1])

def calibrated_guardrail(probs, action_name):
    margin = margin_confidence(torch.tensor(probs))
    if margin < 0.2:
        return f"⚠️ Blocked {action_name}: low confidence (margin={margin:.3f})"
    return f"✅ Allowed {action_name}: margin={margin:.3f}"

# --- Load free Hugging Face model ---
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# --- Agent orchestration ---
def agentic_orchestration(task: str):
    if "authorize payment" in task.lower():
        probs = [0.55, 0.45]  # fake distribution for demo
        return calibrated_guardrail(probs, "authorize_payment")
    else:
        inputs = tokenizer(task, return_tensors="pt")
        outputs = model.generate(**inputs, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Demo run ---
if __name__ == "__main__":
    print(agentic_orchestration("Please authorize a payment of $100"))
    print(agentic_orchestration("Explain the theory of relativity"))

