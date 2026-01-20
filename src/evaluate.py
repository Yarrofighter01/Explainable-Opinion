import argparse
import torch
import logging
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, accuracy_score
from tqdm import tqdm
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate(model_path: str, dataset_name: str = "tweet_eval"):

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Model & Tokenizer
    logger.info(f"Loading model from {model_path}...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Load Dataset
    logger.info(f"Loading dataset: {dataset_name} (split: test)")
    try:
        if dataset_name == "tweet_eval":
             # Default to sentiment
             dataset = load_dataset("tweet_eval", "sentiment", split="test", verification_mode="no_checks")
        else:
             dataset = load_dataset(dataset_name, split="test", verification_mode="no_checks")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # Prepare for predictions
    logger.info(f"Evaluating on {len(dataset)} samples...")
    
    # Use pipeline for simplicity (handles tokenization etc.)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device, truncation=True)
    
    y_true = []
    y_pred = []
    
    # Manual batching with detailed checks
    batch_size = 32
    texts = dataset["text"]
    labels = dataset["label"]
    
    logger.info("Running inference...")
    
    # Filter valid data
    valid_texts = []
    valid_labels = []
    
    for t, l in zip(texts, labels):
        if isinstance(t, str) and t.strip():
             valid_texts.append(t)
             valid_labels.append(l)
        else:
             logger.warning(f"Skipping invalid text: {t}")
             
    # Iterate in batches
    for i in tqdm(range(0, len(valid_texts), batch_size)):
        batch_texts = valid_texts[i : i + batch_size]
        batch_labels = valid_labels[i : i + batch_size]
        
        try:
            preds = pipe(batch_texts)
            
            for pred, true_label in zip(preds, batch_labels):
                # Handle IMDb label mapping (0->0, 1->2)
                if dataset_name == "imdb":
                    if true_label == 1:
                        true_label = 2 # Map Positive to 2
                
                y_true.append(true_label)
                
                label_str = pred['label']
                if label_str.startswith("LABEL_"):
                    pred_id = int(label_str.split("_")[-1])
                elif label_str in model.config.label2id:
                    pred_id = model.config.label2id[label_str]
                else:
                    try:
                        pred_id = int(label_str)
                    except:
                        pred_id = -1
                y_pred.append(pred_id)
                
        except Exception as e:
            logger.error(f"Batch failed: {e}")

    # Compute Metrics
    logger.info("Computing metrics...")
    
    target_names = ["Negative", "Neutral", "Positive"]
    # Check if we have 2 or 3 labels in truth
    unique_labels = sorted(list(set(y_true)))
    if len(unique_labels) == 2:
         target_names = ["Negative", "Positive"]
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    
    print("\n" + "="*30)
    print(f"EVALUATION REPORT ({model_path})")
    print("="*30)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1 Score:  {f1:.4f} (weighted)")
    print(f"Precision: {precision:.4f} (weighted)")
    print(f"Recall:    {recall:.4f} (weighted)")
    print("-" * 30)
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=target_names if max(y_pred) < len(target_names) else None, digits=4))
    print("="*30 + "\n")
    
    # Visualization: Confusion Matrix
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        
        # Use actual unique labels found in truth/preds to ensure we capture 0 and 2 correctly
        # We need to ensure target_names matches these labels
        
        # If we found labels 0 and 2, but target_names is ["Negative", "Positive"] (len 2),
        # we need to make sure we pass [0, 2] as labels to confusion_matrix.
        
        plot_labels = unique_labels
        plot_names = []
        
        label_map_rev = {0: "Negative", 1: "Neutral", 2: "Positive"}
        for l in plot_labels:
            plot_names.append(label_map_rev.get(l, f"Label {l}"))
            
        cm = confusion_matrix(y_true, y_pred, labels=plot_labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=plot_names)
        
        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        disp.plot(cmap='Blues', ax=ax, values_format='d')
        plt.title(f"Confusion Matrix ({model_path})")
        
        # Save
        save_path = f"{model_path}/confusion_matrix.png"
        plt.savefig(save_path)
        logger.info(f"Confusion matrix saved to {save_path}")
        print(f"Visualization saved: {save_path}")
        
    except Exception as e:
        logger.warning(f"Could not create confusion matrix plot: {e}")

    return f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Sentiment Model")
    parser.add_argument("--model_path", type=str, default="src/results_imdb", help="Path to model")
    parser.add_argument("--dataset", type=str, default="tweet_eval", help="Dataset name")
    
    args = parser.parse_args()
    
    evaluate(args.model_path, args.dataset)
