import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from transformers.pipelines import pipeline
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
import gradio as gr
from span_marker import SpanMarkerModel
from fire import Fire

class SafetyInference:

    def __init__(self) -> None:
        
        classifier_model_name = "SamLowe/roberta-base-go_emotions"
        print(f"Loading classifier model {classifier_model_name}")
        self.safety_classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None, device="cuda")

        print(f"Loading NER model")
        self.ner_model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
        self.ner_model = self.ner_model.cuda()
    
    def predict(self, text):
        with torch.no_grad():
            safety_outputs = self.safety_classifier(text)
            ner_predictions = self.ner_model.predict(text)
        breakpoint()
        has_persons = False
        for span in ner_predictions:
            if "person" in span["label"]: # type: ignore
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
        interface = gr.Interface(fn=predict_labels,
                                inputs=gr.Textbox(lines=2, placeholder="Enter Text Here..."),
                                outputs="text",
                                title="NSFW classifier demo",
                                description="Enter some text to see the predicted labels.")

        interface.launch(share=True)
    else:
        res = predict_labels("Joe Biden nude wearing a pink dress")
        print(res)



if __name__ == "__main__":
    Fire(main)
