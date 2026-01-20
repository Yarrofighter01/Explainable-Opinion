import argparse
import logging
from datasets import Dataset
from model_trainer import SentimentTrainer
from data_loader import load_data, preprocess_function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Train Sentiment Analysis Model")
    parser.add_argument("--dataset", type=str, default="imdb", help="Dataset name (imdb, go_emotions, tweet_eval)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to use for training (for debugging)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--model_path", type=str, default="distilbert-base-uncased", help="Path to pretrained model or model identifier")
    parser.add_argument("--shard_index", type=int, default=None, help="Index of the shard to train on (0-based)")
    parser.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--output_dir", type=str, default="./results", help="Directory to save results")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proportion of dataset to include in the test split")
    
    args = parser.parse_args()
    
    logger.info(f"Starting training pipeline with args: {args}")
    
    # Load Data
    num_labels = 2
    try:
        # We load the 'train' split
        dataset = load_data(args.dataset, split="train")
        
        # Detect num_labels immediately
        try:
            if "label" in dataset.features:
                 label_feature = dataset.features["label"]
                 if hasattr(label_feature, "num_classes"):
                     num_labels = label_feature.num_classes
            logger.info(f"Detected {num_labels} labels from dataset features")
        except Exception as e:
             logger.warning(f"Could not detect num_labels: {e}")
        
        # Override for IMDb: We are mapping 2 classes into a 3-class model (0 & 2)
        # So we MUST initialize the model with 3 labels, otherwise dimensions mismatch.
        if args.dataset == "imdb":
            logger.info("Forcing num_labels=3 for IMDb to match pretrained 3-class model.")
            num_labels = 3
        
        if args.samples:
            logger.info(f"Subsetting dataset to {args.samples} samples")
            dataset = dataset.select(range(min(args.samples, len(dataset))))
            
        # Sharding Logic
        if args.shard_index is not None and args.num_shards > 1:
            # IMPORTANT: Shuffle before sharding because IMDb is sorted by label!
            logger.info("Shuffling dataset before sharding (seed=42)...")
            dataset = dataset.shuffle(seed=42)
            
            logger.info(f"Sharding dataset: Use shard {args.shard_index}/{args.num_shards}")
            dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_index)
            logger.info(f"Shard size: {len(dataset)} samples")
            
    except Exception as e:
        logger.error(f"Failed to load/process dataset: {e}")
        return

    # Preprocess
    logger.info("Preprocessing data...")
    # Label Mapping for IMDb (1 -> 2)
    if args.dataset == "imdb":
        logger.info("Mapping IMDb labels: 1 (Positive) -> 2 (Positive)")
        # Cast to integer first to avoid ClassLabel validation error (max 2 classes)
        from datasets import Value
        dataset = dataset.cast_column("label", Value("int32"))
        
        def map_labels(example):
            if example["label"] == 1:
                example["label"] = 2
            return example
        dataset = dataset.map(map_labels)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    
    # Train/Test Split
    logger.info(f"Splitting data with test_size={args.test_size}")
    dataset_dict = tokenized_dataset.train_test_split(test_size=args.test_size)
    train_dataset = dataset_dict["train"]
    eval_dataset = dataset_dict["test"]
    
    # Initialize Trainer
    logger.info(f"Initializing model from {args.model_path} with {num_labels} labels")
    trainer = SentimentTrainer(model_name=args.model_path, num_labels=num_labels)
    
    # Train
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=args.output_dir,
        epochs=args.epochs
    )
    
    logger.info("Pipeline complete.")

if __name__ == "__main__":
    main()
