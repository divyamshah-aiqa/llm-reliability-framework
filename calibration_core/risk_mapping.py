
HIGH_RISK_ACTIONS = [
    "authorize payment",
    "transfer money",
    "delete account",
]

def detect_risk(prompt: str):

    prompt = prompt.lower()

    for action in HIGH_RISK_ACTIONS:
        if action in prompt:
            return "HIGH_RISK"

    return "LOW_RISK"
