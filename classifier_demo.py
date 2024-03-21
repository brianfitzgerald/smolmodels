import torch
from transformers.pipelines import pipeline
import gradio as gr
from span_marker import SpanMarkerModel
from fire import Fire
from typing import List, Dict

from synthetic_data.labels import ANNOTATED_LABELS

safety_labels = {"safe": 0, "unsafe": 1, "borderline": 2}

class SafetyInference:

    def __init__(self) -> None:

        classifier_model_name = "/weka/home-brianf/smolmodels_checkpoints/safety_classifier_binary-Vhe1.hf"
        print(f"Loading classifier model from {classifier_model_name}")
        self.safety_classifier = pipeline(
            task="text-classification",
            model=classifier_model_name,
            top_k=None,
            device="cuda",
        )

        self.safety_classifier.model.config.label2id = safety_labels # type: ignore
        self.safety_classifier.model.config.id2label = {v: k for k, v in safety_labels.items()} # type: ignore
        print(f"Loading NER model")
        self.ner_model = SpanMarkerModel.from_pretrained(
            "tomaarsen/span-marker-mbert-base-multinerd"
        )
        self.ner_model = self.ner_model.cuda()

    def predict(self, text):
        with torch.no_grad():
            safety_outputs: List[List[Dict]] = self.safety_classifier(text)
            ner_predictions: List[Dict] = self.ner_model.predict(text) # type: ignore
        safety_labels_output = ", ".join([f"{out['label']}: {out['score']:.2f}" for out in safety_outputs[0]])
        ner_labels_output = ", ".join([f"{out['span']}: {out['label']} {out['score']:.2f}" for out in ner_predictions])
        has_persons, would_be_nsfw = False, False
        for span in ner_predictions:
            label: str = span["label"]  # type: ignore
            score: float = span["score"]  # type: ignore
            if label == "PER" and score > 0.5:
                has_persons = True
                break
        for out in safety_outputs[0]: # type: ignore
            if out["label"] == "unsafe":
                if out["score"] > 0.5 or (has_persons and score > 0.3):
                    would_be_nsfw = True
                    break
        return safety_labels_output, ner_labels_output, would_be_nsfw


def main(gradio: bool = False):
    inference_wrapper = SafetyInference()

    def predict_labels(text):
        print(f"Processing prompt {text}...")
        safety_outputs, has_persons, would_be_nsfw = inference_wrapper.predict(text)
        return f"Safety classifier labels: {safety_outputs}\nPOS tagger tags: {has_persons}\nWould be NSFW: {would_be_nsfw}"

    if gradio:
        interface = gr.Interface(
            fn=predict_labels,
            inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),
            outputs="text",
            title="NSFW classifier demo",
            description="Enter some text to see the predicted labels.",
        )

        interface.launch(share=True)
    else:
        res = predict_labels("Joe Biden nude wearing a pink dress")
        print(res)


if __name__ == "__main__":
    Fire(main)
