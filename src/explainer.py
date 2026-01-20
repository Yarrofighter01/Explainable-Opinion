import torch
import shap
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from captum.attr import LayerIntegratedGradients, visualization

class ModelExplainer:
    def __init__(self, model_path: str = "distilbert-base-uncased", device=None):

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = device
            
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Pipeline for SHAP
        self.pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer, return_all_scores=True, device=self.device)

    def get_shap_values(self, text: str):

        # SHAP Explainer
        # Using a generic masker
        explainer = shap.Explainer(self.pipeline)
        shap_values = explainer([text])
        return shap_values

    def get_integrated_gradients(self, text: str, target: int = None):

        # Encode inputs
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Prepare for Captum
        def predict_func(inputs, attention_mask=None):
            outputs = self.model(inputs, attention_mask=attention_mask)
            return outputs.logits

        # Integrated Gradients on the embeddings
        # We access the embeddings directly. Note: This depends on model architecture (DistilBert)
        lig = LayerIntegratedGradients(predict_func, self.model.distilbert.embeddings)

        # Baseline (pad token)
        ref_input_ids = input_ids.clone()
        ref_input_ids[0, 1:-1] = self.tokenizer.pad_token_id # Keep CLS and SEP, mask rest
        
        # Predict target class (max logit) if not specific
        logits = predict_func(input_ids, attention_mask)
        if target is None:
            target_class_index = torch.argmax(logits, dim=1).item()
        else:
            target_class_index = int(target)

        # Attribute
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            additional_forward_args=(attention_mask,),
            target=target_class_index,
            return_convergence_delta=True
        )

        # Sum attributions across embedding dimension
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        
        # Create visualization record
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        
        vis_record = visualization.VisualizationDataRecord(
            attributions,
            torch.max(torch.softmax(logits, dim=1)),
            torch.argmax(logits),
            torch.argmax(logits), # Assuming consistent label mapping for now
            str(target_class_index),
            attributions.sum(),
            tokens,
            delta
        )
        
        return vis_record



if __name__ == "__main__":
    # Test
    try:
        explainer = ModelExplainer()
        print(f"Explainer initialized on {explainer.device}")
        text = "I love this movie!"
        # Just checking if methods run without error
        try:
           # shap_vals = explainer.get_shap_values(text) # SHAP can be slow to init
           pass
        except Exception as e:
           print(f"SHAP error: {e}")
           
        try:
           # ig_vals = explainer.get_integrated_gradients(text)
           pass
        except Exception as e:
           print(f"Captum error: {e}")

    except Exception as e:
        print(f"Explainer init failed: {e}")
