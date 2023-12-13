from metaflow import Parameter
from .preprocessing import concatenate_text_fields

TOKENIZER = None

class InferencePLMFlow:
    
    test_s3 = Parameter(
        "test-s3",
        help="S3 path to test data",
        type=str,
        required=True,
    )
    
    model_s3 = Parameter(
        "model-s3",
        help="S3 path to model checkpoint",
        type=str,
        required=True,
    )
    
    output_s3 = Parameter(
        "output-s3",
        help="S3 path to save output predictions",
        type=str,
        required=True,
    )
    
    model_checkpoint = Parameter(
        "model-checkpoint",
        help="The model checkpoint to load",
        type=str,
        required=False,
        default="bert-base-uncased"
    )
    
    def get_tokenizer(self):
        global TOKENIZER
        
        if not TOKENIZER:
            from transformers import AutoTokenizer
            TOKENIZER = AutoTokenizer.from_pretrained(self.model_checkpoint)
        return TOKENIZER
    
    def download_data(self):
        from pendulum_ds.pendulum.datastores.s3_client import S3MetaflowClient as s3
        
        test_fp = "data/temp/test.csv"
        
        s3.get_file(self.test_s3, test_fp)
        
        return test_fp
    
    def get_ds(self, max_memory_size=16_000):
        import datasets
        from datasets import load_dataset
        
        datasets.config.IN_MEMORY_MAX_SIZE = max_memory_size
        
        test_fp = self.download_data()
        dataset = load_dataset("csv", data_files=test_fp, split="train", keep_in_memory=True)
        
        return dataset
    
    def preprocess_function(self, batch):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(batch["text"], truncation=True)
        return encoding
    
    def preprocess_ds(self, dataset, text_cols=None):
        import torch
        
        if text_cols is not None:
            dataset = dataset.map(lambda batch: concatenate_text_fields(batch, text_cols), batched=True, remove_columns=text_cols)
            
        dataset = dataset.map(self.preprocess_function, batched=True)
        
        format = {'type': 'torch', 'format_kwargs' :{'dtype': torch.long}}
        dataset.set_format(**format)
        
        return dataset
    
    def get_model(self):
        import os
        from pendulum_ds.pendulum.datastores.s3_client import S3MetaflowClient as s3
        
        if self.model_s3.endswith("/"):
            model_name = os.path.basename(self.model_s3[:-1])
        else:
            model_name = os.path.basename(self.model_s3)
            
        model_dir = f"models/{model_name}"
        
        s3.get_dir(self.model_s3, model_dir)
        
        return os.path.abspath(model_dir)
    