from metaflow import Parameter
from .preprocessing import concatenate_text_fields

TOKENIZER = None

class FinetunePLMFlow:

    train_s3 = Parameter(
        "train-s3",
        help="S3 path to training data",
        type=str,
        required=True,
    )
    
    val_s3 = Parameter(
        "val-s3",
        help="S3 path to validation data",
        type=str,
        required=True,
    )
    
    outdir_s3 = Parameter(
        "outdir-s3",
        help="S3 path to output directory",
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
    
    batch_size = Parameter(
        "batch-size",
        help="The batch size to use",
        type=int,
        required=False,
        default=16
    )
    
    learning_rate = Parameter(
        "learning-rate",
        help="The learning rate to use",
        type=float,
        required=False,
        default=2e-5
    )
    
    weight_decay = Parameter(
        "weight-decay",
        help="The weight decay to use",
        type=float,
        required=False,
        default=0.01
    )
    
    epochs = Parameter(
        "epochs",
        help="The number of epochs to train",
        type=int,
        required=False,
        default=3
    )
    
    report_wandb = Parameter(
        "report-wandb",
        help="Whether to report to wandb",
        type=bool,
        required=False,
        default=False
    )
        
    def get_tokenizer(self):
        global TOKENIZER
        
        if not TOKENIZER:
            from transformers import AutoTokenizer
            TOKENIZER = AutoTokenizer.from_pretrained(self.model_checkpoint)
        return TOKENIZER
    
    def download_data(self):
        from pendulum_ds.pendulum.datastores.s3_client import S3MetaflowClient as s3
        
        train_fp = "data/temp/train.csv"
        val_fp = "data/temp/val.csv"
        
        s3.get_file(self.train_s3, train_fp)
        s3.get_file(self.val_s3, val_fp)
        
        return train_fp, val_fp
    
    def get_ds(self, max_memory_size=16_000):
        import datasets
        from datasets import load_dataset
        
        datasets.config.IN_MEMORY_MAX_SIZE = max_memory_size
        
        train_fp, val_fp = self.download_data()
        dataset = load_dataset("csv", data_files={"train": train_fp, "val": val_fp}, keep_in_memory=True)
        
        return dataset
    
    def preprocess_function(self, batch, label_type):
        tokenizer = self.get_tokenizer()
        encoding = tokenizer(batch["text"], truncation=True)
        
        if label_type == "multi":
            import numpy as np
            
            labels_matrix = np.zeros((len(batch), len(self.labels)))
            for label in self.labels:
                labels_matrix[:, self.label2id[label]] = batch[label]
            
            encoding["labels"] = labels_matrix.tolist()
        
        return encoding
    
    def preprocess_ds(self, dataset, text_cols=None, label_type="binary"):
        import torch
        
        if text_cols is not None:
            dataset = dataset.map(lambda batch: concatenate_text_fields(batch, text_cols), batched=True, remove_columns=text_cols)
            
        dataset = dataset.map(lambda x: self.preprocess_function(x, label_type), batched=True, remove_columns=[col for col in dataset['train'].column_names if col != 'label'])

        format = {'type': 'torch', 'format_kwargs' :{'dtype': torch.long}}
        dataset.set_format(**format)
        
        return dataset
    
    def train_model(self, compute_metrics):
        import os
        from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
        
        data_collator = DataCollatorWithPadding(tokenizer=self.get_tokenizer())
        
        tokenizer = self.get_tokenizer()   
        out_dir = f"{os.path.basename(self.train_s3).replace('.train.csv', '')}/{self.model_checkpoint}"
        
        args = TrainingArguments(
            output_dir=out_dir,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            num_train_epochs=self.epochs,
            weight_decay=self.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            report_to="wandb" if self.report_wandb else "none"
        )
        
        trainer = Trainer(
            self.model,
            args,
            train_dataset=self.tokenized_dataset["train"].shuffle(seed=42),
            eval_dataset=self.tokenized_dataset["val"].shuffle(seed=42),
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        
        train_results = trainer.train()
        trainer.save_metrics('train', train_results.metrics)
        
        metrics = trainer.evaluate(self.tokenized_dataset["val"])
        trainer.save_metrics('eval', metrics)
        
        trainer.save_model()
        return out_dir
        
    @staticmethod
    def save_model(outdir_s3, out_dir):
        from pendulum_ds.pendulum.datastores.s3_client import S3MetaflowClient as s3
        
        s3.put_dir(outdir_s3, out_dir)
        
    @staticmethod
    def get_model(outdir_s3, out_dir):
        from pendulum_ds.pendulum.datastores.s3_client import S3MetaflowClient as s3
        
        s3.get_dir(outdir_s3, out_dir)