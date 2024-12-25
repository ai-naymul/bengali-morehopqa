from datasets import load_dataset, Dataset
from huggingface_hub import hf_hub_download
from googletrans import Translator
import json
import logging

class BengaliDatasetTranslator:
    def __init__(self, hf_token):
        self.token = hf_token
        self.translator = Translator()
        
    def translate_to_bengali(self, text):
        """Translate text to Bengali."""
        try:
            return self.translator.translate(text, dest='bn').text
        except:
            return text
            
    def translate_decomposition(self, decomp_list):
        """Translate question decomposition items."""
        translated = []
        for item in decomp_list:
            translated_item = {
                'sub_id': item['sub_id'],
                'question': self.translate_to_bengali(item['question']),
                'answer': self.translate_to_bengali(item['answer']),
                'paragraph_support_title': self.translate_to_bengali(item['paragraph_support_title'])
            }
            translated.append(translated_item)
        return translated
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

    def translate_dataset(self, dataset):
        """Translate the entire dataset to Bengali."""
        translated_data = []
        
        for item in dataset:
            translated_item = {
                'question': self.translate_to_bengali(item['question']),
                'answer': self.translate_to_bengali(item['answer']),
                'context': self.translate_to_bengali(item['context']),
                'previous_question': self.translate_to_bengali(item['previous_question']),
                'previous_answer': self.translate_to_bengali(item['previous_answer']),
                'question_decomposition': self.translate_decomposition(item['question_decomposition']),
                # 'question_on_last_hop': self.translate_to_bengali(item['question_on_last_hop']),
                'answer_type': item['answer_type'],
                'previous_answer_type': item['previous_answer_type'],
                'no_of_hops': item['no_of_hops'],
                'reasoning_type': item['reasoning_type']
            }
            print(translated_item)
            translated_data.append(translated_item)
            
        return Dataset.from_list(translated_data)
    
    def push_to_hub(self, dataset, repo_name):
        """Push the translated dataset to Hugging Face Hub."""
        dataset.push_to_hub(repo_name, token=self.token)

# Usage
translator = BengaliDatasetTranslator('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')
original_dataset = translator.download_dataset()
bengali_dataset = translator.translate_dataset(original_dataset)
translator.push_to_hub(bengali_dataset, 'nymtheescobar/bengali-morehopqa')
