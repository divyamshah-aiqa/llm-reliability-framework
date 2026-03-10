import json
from datetime import datetime

LOG_FILE = "reliability_logs.json"

def log_event(prompt, risk_level, confidence_margin, decision):

    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "prompt": prompt,
        "risk_level": risk_level,
        "confidence_margin": confidence_margin,
        "decision": decision
    }

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)
    except:
        logs = []

    logs.append(event)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2)
