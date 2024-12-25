from datasets import load_dataset, Dataset
from googletrans import Translator
from tqdm import tqdm
import time
from huggingface_hub import HfApi, login, hf_hub_download
import os
import json
import logging
from typing import List, Dict, Any
import backoff

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation.log'),
        logging.StreamHandler()
    ]
)

# Constants
HUGGINGFACE_TOKEN = "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"  # Replace with your token
CHECKPOINT_INTERVAL = 100
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 0.5
RETRY_DELAY = 2

class DatasetTranslator:
    def __init__(self, token: str):
        self.token = token
        self.translator = None
        self.setup_authentication()
        
    def setup_authentication(self) -> None:
        """Setup authentication with Hugging Face."""
        try:
            login(token=self.token, add_to_git_credential=True)
            logging.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logging.error(f"Authentication error: {e}")
            raise

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def create_translator(self) -> None:
        """Create Google Translator instance with retry mechanism."""
        self.translator = Translator(service_urls=['translate.google.com'])
        
    def translate_text(self, text: str) -> str:
        """Translate text to Bengali with retry mechanism."""
        if not text:
            return ""
            
        for attempt in range(MAX_RETRIES):
            try:
                result = self.translator.translate(text, dest='bn')
                time.sleep(RATE_LIMIT_DELAY)
                return result.text
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logging.info(f"Here is the texts: {text[:50]}")
                    logging.warning(f"Failed to translate: {text[:50]}... Error: {e}")
                    return text
                time.sleep(RETRY_DELAY)
                continue

    def translate_question_decomposition(self, decomp_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate question decomposition components."""
        if not decomp_list:
            return []
        
        translated_decomp = []
        for item in decomp_list:
            translated_item = {
                "sub_id": item["sub_id"],
                "question": self.translate_text(item["question"]),
                "answer": self.translate_text(item["answer"]),
                "paragraph_support_title": self.translate_text(item["paragraph_support_title"])
            }
            translated_decomp.append(translated_item)
        return translated_decomp

    def download_dataset(self) -> Dataset:
        """Download and load the dataset from Hugging Face."""
        try:
            file_path = hf_hub_download(
                repo_id="alabnii/morehopqa",
                filename="data/with_human_verification.json",
                repo_type="dataset",
                token=self.token
            )
            
            logging.info(f"Dataset downloaded to: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Convert nested lists to strings before creating dataset
            processed_data = []
            for item in data:
                processed_item = {}
                for key, value in item.items():
                    if isinstance(value, list):
                        if key == "question_decomposition":
                            # Handle question_decomposition specially
                            processed_item[key] = [
                                {k: str(v) if isinstance(v, (list, dict)) else v 
                                for k, v in d.items()}
                                for d in value
                            ]
                        else:
                            processed_item[key] = str(value)
                    else:
                        processed_item[key] = value
                processed_data.append(processed_item)
                
            return Dataset.from_list(processed_data)
        
        except Exception as e:
            logging.error(f"Error downloading dataset: {e}")
            raise

    def save_checkpoint(self, translated_data: List[Dict[str, Any]], checkpoint_num: int) -> None:
        """Save translation checkpoint."""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{checkpoint_num}.json")
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def create_dataset_description(self) -> str:
        """Create dataset description for Hugging Face."""
        return """
        # Bengali MoreHopQA Dataset
        
        This is a Bengali translation of the MoreHopQA dataset. The original dataset contains multi-hop questions and answers,
        which have been automatically translated to Bengali using Google Translate.
        
        ## Dataset Structure
        - question: The main question in Bengali
        - answer: The answer in Bengali
        - context: Supporting context in Bengali
        - previous_question: Previous related question in Bengali
        - previous_answer: Previous answer in Bengali
        - question_decomposition: Decomposed sub-questions and answers in Bengali
        - question_on_last_hop: Final hop question in Bengali
        
        ## Fields Maintained from Original Dataset
        - answer_type
        - previous_answer_type
        - no_of_hops
        - reasoning_type
        
        Original dataset: alabnii/morehopqa
        """

    def translate_and_push_dataset(self) -> None:
        """Main function to translate and push dataset to Hugging Face."""
        try:
            logging.info("Starting dataset translation process...")
            
            # Download dataset
            dataset = self.download_dataset()
            
            # Create translator
            self.create_translator()
            
            translated_data = []
            
            # Translation loop
            for idx, item in enumerate(tqdm(dataset, desc="Translating dataset")):
                translated_item = {
                    "question": self.translate_text(item["question"]),
                    "answer": self.translate_text(item["answer"]),
                    "context": self.translate_text(item["context"]),
                    "previous_question": self.translate_text(item["previous_question"]),
                    "previous_answer": self.translate_text(item["previous_answer"]),
                    "question_decomposition": self.translate_question_decomposition(
                        item["question_decomposition"]
                    ),
                    "question_on_last_hop": self.translate_text(item["question_on_last_hop"]),
                    "answer_type": item["answer_type"],
                    "previous_answer_type": item["previous_answer_type"],
                    "no_of_hops": item["no_of_hops"],
                    "reasoning_type": item["reasoning_type"]
                }
                translated_data.append(translated_item)
                
                # Save checkpoint
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    logging.info(f"Progress: {idx + 1} items processed")
                    self.save_checkpoint(translated_data, idx + 1)
            
            # Create final dataset
            logging.info("Creating Hugging Face dataset...")
            bn_dataset = Dataset.from_list(translated_data)
            
            # Create dataset card
            dataset_card = self.create_dataset_description()
            
            # Push to Hugging Face
            logging.info("Pushing to Hugging Face Hub...")
            bn_dataset.push_to_hub(
                "bengali-morehopqa",
                private=False,
                commit_message="Add Bengali translation of MoreHopQA",
                token=self.token
            )
            
            # Upload dataset card
            api = HfApi()
            api.upload_file(
                path_or_fileobj=dataset_card.encode(),
                path_in_repo="README.md",
                repo_id="bengali-morehopqa",
                token=self.token
            )
            
            logging.info("Dataset successfully pushed to Hugging Face!")
            
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

def main():
    """Main execution function."""
    try:
        translator = DatasetTranslator(HUGGINGFACE_TOKEN)
        translator.translate_and_push_dataset()
    except Exception as e:
        logging.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
