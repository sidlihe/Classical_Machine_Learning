"""Small Gradio UI to test saved models on sample input rows.

Usage:
  python scripts/gradio_app.py --models_dir models

The app looks for pickled sklearn-like models in `models/` and exposes a simple
input JSON field where users can paste a single-row dict of features.
"""
import argparse
import os
import glob
import json
import pickle
from pathlib import Path

import gradio as gr
import pandas as pd


def load_models(models_dir: str):
    models = {}
    p = Path(models_dir)
    for fp in p.glob("**/*.pkl"):
        name = fp.stem
        try:
            with open(fp, "rb") as f:
                model = pickle.load(f)
            models[name] = model
        except Exception:
            # ignore non-pickle or incompatible files
            continue
    return models


def predict_with_model(model, input_json: str):
    try:
        data = json.loads(input_json)
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            return "Input JSON must be an object or list of objects"
    except Exception as e:
        return f"Invalid JSON: {e}"

    try:
        # If model has predict_proba, include probabilities
        pred = model.predict(df)
        proba = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df).tolist()

        out = {"prediction": pred.tolist() if hasattr(pred, 'tolist') else pred, "probabilities": proba}
        return json.dumps(out, indent=2)
    except Exception as e:
        return f"Model prediction failed: {e}"


def build_ui(models_dir: str):
    models = load_models(models_dir)
    model_names = list(models.keys())

    with gr.Blocks() as demo:
        gr.Markdown("# Model tester\nPaste a single-row JSON of features and pick a model to run.")

        with gr.Row():
            model_dropdown = gr.Dropdown(choices=model_names, label="Model", value=model_names[0] if model_names else None)
            refresh_btn = gr.Button("Refresh models")

        input_box = gr.Textbox(label="Input JSON (object or list)", lines=8, value='{}')
        run_btn = gr.Button("Run")
        output_box = gr.Textbox(label="Output", lines=12)

        def run(model_name, input_json):
            if model_name not in models:
                return "Model not found. Click Refresh models."
            model = models[model_name]
            return predict_with_model(model, input_json)

        def refresh():
            new_models = load_models(models_dir)
            models.clear()
            models.update(new_models)
            return gr.Dropdown.update(choices=list(models.keys()), value=list(models.keys())[0] if models else None)

        run_btn.click(run, inputs=[model_dropdown, input_box], outputs=output_box)
        refresh_btn.click(refresh, outputs=model_dropdown)

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", default="models", help="Directory with saved model .pkl files")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    demo = build_ui(args.models_dir)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
