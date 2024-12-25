import logging
from datasets import Dataset
from huggingface_hub import hf_hub_download
from googletrans import Translator
import json
from typing import Union, List, Dict

class DatasetTranslator:
    def __init__(self, token: str):
        self.token = token
        self.translator = Translator()
        
    def translate_text(self, text: str) -> str:
        """Translate a single text string to Bengali."""
        try:
            return self.translator.translate(text, dest='bn').text
        except Exception as e:
            logging.error(f"Translation error: {e}")
            return text
            
    def translate_list(self, items: List[str]) -> List[str]:
        """Translate a list of strings."""
        return [self.translate_text(item) for item in items]
    
    def translate_question_decomposition(self, decomp_list: List[Dict]) -> List[Dict]:
        """Translate the question decomposition structure."""
        translated_decomp = []
        for item in decomp_list:
            translated_item = {
                'sub_id': item['sub_id'],
                'question': self.translate_text(item['question']),
                'answer': self.translate_text(item['answer']),
                'paragraph_support_title': self.translate_text(item['paragraph_support_title'])
            }
            translated_decomp.append(translated_item)
        return translated_decomp
    
    def translate_context(self, context: Dict) -> Dict:
        """Translate the context structure."""
        return {
            'title': self.translate_text(context['title']),
            'paragraphs': self.translate_list(context['paragraphs'])
        }
    
    def translate_item(self, item: Dict) -> Dict:
        """Translate a single data item with all its fields."""
        translated_item = {}
        for key, value in item.items():
            if key == 'question_decomposition':
                translated_item[key] = self.translate_question_decomposition(value)
            elif key == 'context':
                translated_item[key] = self.translate_context(value)
            elif isinstance(value, str):
                translated_item[key] = self.translate_text(value)
            else:
                translated_item[key] = value
        return translated_item

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
            
            translated_data = [self.translate_item(item) for item in data]
            return Dataset.from_list(translated_data)
            
        except Exception as e:
            logging.error(f"Error downloading dataset: {e}")
            raise
            
    def push_to_hub(self, dataset: Dataset, repo_name: str):
        """Push the translated dataset to Hugging Face Hub."""
        try:
            dataset.push_to_hub(
                repo_name,
                token=self.token,
                private=True
            )
            logging.info(f"Dataset successfully pushed to {repo_name}")
        except Exception as e:
            logging.error(f"Error pushing to hub: {e}")
            raise

def main():
    # Initialize logging
    logging.basicConfig(level=logging.INFO)
    
    # Your HuggingFace token
    token = "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"
    
    # Initialize translator
    translator = DatasetTranslator(token)
    
    # Download and translate dataset
    dataset = translator.download_dataset()
    
    # Push to HuggingFace
    translator.push_to_hub(dataset, "your-username/bengali-morehopqa")

if __name__ == "__main__":
    main()
