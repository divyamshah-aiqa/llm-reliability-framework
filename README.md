---
title: LLM Reliability Evaluation Framework
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
---


LLM Reliability Evaluation Framework

A confidence-calibrated decision gating system that evaluates whether an AI should:

Respond

Ask for clarification

Defer

Remain silent

before invoking an LLM.

🔍 What This Project Does

Instead of blindly generating responses, this system first decides:

Should the AI act at all?

It uses semantic embeddings + a trained classifier to output:

Decision

Confidence score

Decision margin

Risk category

🏗 System Architecture

Pipeline:

User Input
→ SentenceTransformer Embedding (MiniLM-L6-v2)
→ 3-layer PyTorch classifier
→ Softmax confidence + margin
→ Calibration logic
→ Final decision

📊 Output Signals

Confidence → Probability of predicted class

Decision Margin → Gap between top-2 probabilities

Risk Category → trusted / ambiguous / high_risk / noise

🧠 Calibration Logic

Rules applied on top of raw model prediction:

Low confidence → silent

Low margin → ask_clarify

Empty input → silent

High confidence + strong margin → trusted

This ensures controlled AI behavior.

🚀 Deployment

Built with PyTorch + SentenceTransformers

Deployed on Hugging Face Spaces (Gradio UI)

Fully interactive reliability evaluation

🎯 Why It Matters

Most AI demos generate text.

This project measures and controls when generation should happen.

Focus:
✔ Reliability
✔ Confidence gating
✔ Safe AI activation
✔ Deployable evaluation system