from datasets import load_dataset, Dataset
from googletrans import Translator
from tqdm import tqdm
import time
from huggingface_hub import HfApi, login, hf_hub_download
import os
import json
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_log.txt'),
        logging.StreamHandler()
    ]
)

# Constants
HUGGINGFACE_TOKEN = "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"  # Replace with your token
CHECKPOINT_INTERVAL = 50
TRANSLATION_DELAY = 0.5
MAX_RETRIES = 3
RETRY_DELAY = 2

class DatasetTranslator:
    def __init__(self):
        self.api = HfApi()
        self.translator = None
        self.setup_authentication()
        
    def setup_authentication(self):
        """Setup authentication with Hugging Face"""
        try:
            login(token=HUGGINGFACE_TOKEN, add_to_git_credential=True)
            logging.info("Successfully authenticated with Hugging Face")
        except Exception as e:
            logging.error(f"Authentication failed: {e}")
            raise

    def create_translator(self):
        """Create Google Translator instance"""
        if not self.translator:
            self.translator = Translator(service_urls=['translate.google.com'])
        return self.translator

    def translate_text(self, text):
        """Translate text with retry mechanism"""
        if not text:
            return ""
            
        for attempt in range(MAX_RETRIES):
            try:
                result = self.translator.translate(text, dest='bn')
                time.sleep(TRANSLATION_DELAY)
                return result.text
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    logging.warning(f"Translation failed after {MAX_RETRIES} attempts for: {text[:50]}...")
                    return text
                time.sleep(RETRY_DELAY)
                continue

    def translate_question_decomposition(self, decomp_list):
        """Translate question decomposition items"""
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

    def download_dataset(self):
        """Download and load the dataset"""
        try:
            logging.info("Downloading dataset...")
            file_path = hf_hub_download(
                repo_id="alabnii/morehopqa",
                filename="data/with_human_verification.json",
                repo_type="dataset",
                token=HUGGINGFACE_TOKEN
            )
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            dataset = Dataset.from_list(data)
            logging.info(f"Dataset loaded successfully. Size: {len(dataset)} items")
            return dataset
            
        except Exception as e:
            logging.error(f"Failed to download dataset: {e}")
            raise

    def save_checkpoint(self, translated_data, checkpoint_num):
        """Save translation checkpoint"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(
            checkpoint_dir, 
            f"checkpoint_{checkpoint_num}_{timestamp}.json"
        )
        
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def create_dataset_description(self):
        """Create dataset documentation"""
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
        - answer_type: Type of the expected answer (unchanged)
        - previous_answer_type: Type of the answer to the previous question (unchanged)
        - no_of_hops: Number of reasoning hops (unchanged)
        - reasoning_type: Type of reasoning required (unchanged)
        
        Original dataset: alabnii/morehopqa
        """

    def translate_and_push_dataset(self):
        """Main function to translate and upload dataset"""
        try:
            self.create_translator()
            dataset = self.download_dataset()
            translated_data = []
            
            logging.info("Starting translation process...")
            for idx, item in enumerate(tqdm(dataset)):
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
                
                if (idx + 1) % CHECKPOINT_INTERVAL == 0:
                    logging.info(f"Progress: {idx + 1}/{len(dataset)} items processed")
                    self.save_checkpoint(translated_data, idx + 1)
            
            logging.info("Creating Hugging Face dataset...")
            bn_dataset = Dataset.from_list(translated_data)
            
            logging.info("Pushing to Hugging Face Hub...")
            bn_dataset.push_to_hub(
                "bengali-morehopqa",
                private=False,
                commit_message="Add Bengali translation of MoreHopQA",
                token=HUGGINGFACE_TOKEN
            )
            
            # Upload dataset documentation
            self.api.upload_file(
                path_or_fileobj=self.create_dataset_description().encode(),
                path_in_repo="README.md",
                repo_id="bengali-morehopqa",
                token=HUGGINGFACE_TOKEN
            )
            
            logging.info("Dataset successfully pushed to Hugging Face!")
            
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

def main():
    try:
        translator = DatasetTranslator()
        translator.translate_and_push_dataset()
    except Exception as e:
        logging.error(f"Process failed: {e}")
        raise

if __name__ == "__main__":
    main()
