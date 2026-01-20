import logging
from datasets import load_dataset
from transformers import DistilBertTokenizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(dataset_name: str, split: str = "train"):

    logger.info(f"Loading dataset: {dataset_name} (split: {split})")
    
    # Handle config names (e.g., tweet_eval:sentiment)
    config_name = None
    if ":" in dataset_name:
        dataset_name, config_name = dataset_name.split(":")
    
    try:
        if config_name:
             dataset = load_dataset(dataset_name, config_name, split=split, verification_mode="no_checks")
        elif dataset_name == "tweet_eval":
             # Default to sentiment for this project if not specified
             dataset = load_dataset("tweet_eval", "sentiment", split=split, verification_mode="no_checks")
        else:
            dataset = load_dataset(dataset_name, split=split, verification_mode="no_checks")
            
        return dataset
    except Exception as e:
        logger.error(f"Error loading dataset {dataset_name}: {e}")
        raise

def preprocess_function(examples):

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Handle different column names (imdb uses 'text', go_emotions uses 'text')
    text_column = "text"
    if text_column not in examples:
         # Fallback or check for other common names if needed
         pass

    return tokenizer(examples[text_column], truncation=True, padding="max_length", max_length=128)

if __name__ == "__main__":
    # Simple test
    try:
        data = load_data("imdb", split="train[:1%]")
        print(f"Loaded {len(data)} examples from IMDb.")
    except Exception as e:
        print(f"Failed to load data: {e}")
