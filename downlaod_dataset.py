import logging
from huggingface_hub import hf_hub_download
from datasets import Dataset
import json


class DatasetDownloader():
    def download_dataset(self) -> Dataset:
        """Download and load the dataset from Hugging Face."""
        try:
            file_path = hf_hub_download(
                repo_id="alabnii/morehopqa",
                filename="data/with_human_verification.json", # download the human verified part of the dataset
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