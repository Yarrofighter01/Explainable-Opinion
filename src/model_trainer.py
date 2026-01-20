import logging
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    DistilBertTokenizer
)
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentTrainer:
    def __init__(self, model_name: str = "distilbert-base-uncased", num_labels: int = 2):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model_name = model_name
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    def train(self, train_dataset, eval_dataset=None, output_dir: str = "./results", epochs: int = 1):

        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            dataloader_num_workers=0,
            num_train_epochs=epochs,
            weight_decay=0.01,
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if eval_dataset else False,
        )

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )

        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete.")
        
        # Save the final model
        self.save_model(output_dir)

    def save_model(self, path: str):

        logger.info(f"Saving model to {path}")
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def predict(self, text: str):
         # Helper for quick prediction if needed (mostly for testing trainer)
         inputs = self.tokenizer(text, return_tensors="pt", truncation=True).to(self.device)
         with torch.no_grad():
             logits = self.model(**inputs).logits
         probabilities = torch.nn.functional.softmax(logits, dim=-1)
         return probabilities.cpu().numpy()

if __name__ == "__main__":
    # Test initialization
    trainer = SentimentTrainer()
    print(f"Trainer initialized on {trainer.device}")
