import torch
from transformers.pipelines import pipeline
import gradio as gr
from span_marker import SpanMarkerModel
from fire import Fire

from synthetic_data.labels import ANNOTATED_LABELS

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

        self.safety_classifier.model.config.label2id = ANNOTATED_LABELS # type: ignore
        self.safety_classifier.model.config.id2label = {v: k for k, v in ANNOTATED_LABELS.items()} # type: ignore
        print(f"Loading NER model")
        self.ner_model = SpanMarkerModel.from_pretrained(
            "tomaarsen/span-marker-mbert-base-multinerd"
        )
        self.ner_model = self.ner_model.cuda()

    def predict(self, text):
        breakpoint()
        with torch.no_grad():
            safety_outputs = self.safety_classifier(text)
            ner_predictions = self.ner_model.predict(text)
        has_persons: bool = False
        for span in ner_predictions:
            label: str = span["label"]  # type: ignore
            score: float = span["score"]  # type: ignore
            if label == "PER" and score > 0.5:
                has_persons = True
                break
        return safety_outputs, has_persons


def main(gradio: bool = False):
    inference_wrapper = SafetyInference()

    def predict_labels(text):
        print(f"Processing {text}")
        safety_outputs, has_persons = inference_wrapper.predict(text)
        return f"Safety: {safety_outputs}, has people: {has_persons}"

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
