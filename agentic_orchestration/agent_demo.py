from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from calibration_core.calibration import margin_confidence
from mlops_deployment.logging_system import log_event

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


HIGH_RISK_PATTERNS = [
    "authorize payment",
    "approve payment",
    "approve this payment",
    "transfer money",
    "bypass safety",
]


def is_high_risk(prompt):

    prompt = prompt.lower()

    # keyword match
    for pattern in HIGH_RISK_PATTERNS:
        if pattern in prompt:
            return True

    # semantic heuristic: approve + payment together
    if "approve" in prompt and "payment" in prompt:
        return True

    return False


def agentic_orchestration(task: str):

    if is_high_risk(task):

        probs = [0.55, 0.45]
        margin = margin_confidence(torch.tensor(probs))

        decision = "BLOCKED"

        log_event(task, "HIGH_RISK", margin, decision)

        return f"⚠️ Blocked high-risk action (margin={margin:.3f})"

    inputs = tokenizer(task, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    margin = 0.9
    decision = "ANSWERED"

    log_event(task, "LOW_RISK", margin, decision)

    return answer
