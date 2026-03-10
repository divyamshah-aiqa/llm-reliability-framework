
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from calibration_core.calibration import calibrated_decision
from calibration_core.risk_mapping import detect_risk

model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def agent_system(prompt: str):

    risk = detect_risk(prompt)

    if risk == "HIGH_RISK":
        return "⚠️ Escalation required: High-risk instruction detected."

    probs = [0.55, 0.45]

    decision = calibrated_decision(probs)

    if decision == "LOW_CONFIDENCE":
        return "⚠️ Model uncertain. Asking clarification."

    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=100)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
