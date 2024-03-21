import torch
from transformers.models.roberta.modeling_roberta import (
    RobertaForSequenceClassification,
)
from transformers.models.roberta.tokenization_roberta import RobertaTokenizer
import gradio as gr
from span_marker import SpanMarkerModel
from fire import Fire

class SafetyInference:

    def __init__(self) -> None:
        
        classifier_model_name = "SamLowe/roberta-base-go_emotions"
        print(f"Loading classifier model {classifier_model_name}")
        self.safety_classifier = RobertaForSequenceClassification.from_pretrained(classifier_model_name)
        self.safety_classifier = self.safety_classifier.cuda()
        self.tokenizer = RobertaTokenizer.from_pretrained(classifier_model_name)

        print(f"Loading NER model")
        self.ner_model = SpanMarkerModel.from_pretrained("tomaarsen/span-marker-mbert-base-multinerd")
        self.ner_model = self.ner_model.cuda()
    
    def predict(self, text):
        tokenized_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        with torch.no_grad():
            logits = self.safety_classifier(**tokenized_inputs).logits
            logprobs = torch.sigmoid(logits).cpu().numpy()
            ner_predictions = self.ner_model.predict(text)
        breakpoint()
        has_persons = False
        for span in ner_predictions:
            if "person" in span["label"]: # type: ignore
                has_persons = True
                break
        return logprobs, has_persons

def main(gradio: bool = False):
    inference_wrapper = SafetyInference()

    def predict_labels(text):
        print(f"Processing {text}")
        logprobs, has_persons = inference_wrapper.predict(text)
        return f"Safety: {logprobs}, has people: {has_persons}"

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
