import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset, DatasetDict
import numpy as np
import huggingface_hub
from sklearn.metrics import f1_score
import sys
import gc
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import warnings
import argparse
warnings.filterwarnings('ignore')

token = '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'
huggingface_hub.login(token)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

def prepare_dataset(top_n):
    synthdetoxm_new = load_dataset('alexandro767/synthdetoxm_ru_with_token_classes')['train']
    synthdetoxm_new.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    
    def go_squeeze(examples):
        input_ids = examples['input_ids'].squeeze()
        attention_mask = examples['attention_mask'].squeeze()
        labels = examples['labels'].squeeze()
    
        return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
        
    def align_tokens_with_pad(examples):
        token_classes = examples['token_classes']
        attention_mask = examples['attention_mask']
        
        ignore_value = -100
        ignore_tensor = torch.full_like(token_classes, ignore_value)
        
        token_classes = torch.where(attention_mask == 0, ignore_tensor, token_classes)
    
        return {'token_classes': token_classes}
    
    synthdetoxm_new = synthdetoxm_new.map(go_squeeze, batched=True)
    synthdetoxm_new = synthdetoxm_new.map(align_tokens_with_pad, batched=True)
    
    df_for_collator = Dataset.from_dict({'toxic_sentence': synthdetoxm_new['toxic_sentence'][:top_n],
                                         'neutral_sentence': synthdetoxm_new['neutral_sentence'][:top_n],
                                         'lang': synthdetoxm_new['lang'][:top_n],
                                         'cls_input_ids': synthdetoxm_new['input_ids'][:top_n], 
                                         'cls_attention_mask': synthdetoxm_new['attention_mask'][:top_n], 
                                         'cls_labels': synthdetoxm_new['token_classes'][:top_n], 
                                         'detox_input_ids': synthdetoxm_new['input_ids'][:top_n], 
                                         'detox_attention_mask': synthdetoxm_new['attention_mask'][:top_n],
                                         'detox_labels': synthdetoxm_new['labels'][:top_n]})
    return df_for_collator


class TwoLossesModel(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classification_head = nn.Linear(config.d_model, 3)


class CustomTrainer(Trainer):
    def __init__(self, *args, weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.classification_loss_fn = nn.CrossEntropyLoss()
        self.weight = weight  # Store the weight parameter

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if model.training:
            cls_input_ids = inputs.get("cls_input_ids")
            cls_attention_mask = inputs.get("cls_attention_mask")
            cls_labels = inputs.get("cls_labels")
            detox_input_ids = inputs.get("detox_input_ids")
            detox_attention_mask = inputs.get("detox_attention_mask")
            detox_labels = inputs.get("detox_labels")
        
            detox_loss = torch.tensor(0.0, device=detox_input_ids.device)
            detox_outputs = None
            if detox_labels is not None:
                detox_outputs = model(input_ids=detox_input_ids, attention_mask=detox_attention_mask, labels=detox_labels)
                detox_loss = detox_outputs.loss
                detox_loss = torch.mean(detox_loss, axis=0)
    
            
            classification_loss = torch.tensor(0.0, device=detox_input_ids.device)
            if cls_labels is not None:
                encoder_outputs = model.module.encoder(input_ids=cls_input_ids, attention_mask=cls_attention_mask, return_dict=True)
                hidden_states = encoder_outputs.last_hidden_state
        
                classification_logits = model.module.classification_head(hidden_states)
        
                classification_preds = classification_logits.view(-1, classification_logits.shape[-1])
                classification_labels = cls_labels.view(-1)
                
                classification_loss = self.classification_loss_fn(classification_preds, classification_labels)
        
            if self.weight != 1.:
                total_loss = (1.-self.weight) * detox_loss + self.weight * classification_loss
            else:
                total_loss = detox_loss + classification_loss
            print(detox_loss, classification_loss)
    
            if return_outputs:
                return total_loss, detox_outputs

            del cls_input_ids
            del cls_attention_mask
            del cls_labels
            del detox_input_ids
            del detox_attention_mask
            del detox_labels

            del detox_outputs
            del encoder_outputs
            del hidden_states
            del classification_logits
            del classification_preds
            del classification_labels

            gc.collect()
            torch.cuda.empty_cache()
    
            return total_loss
                
        else:
            detox_input_ids = inputs.get("input_ids")
            detox_attention_mask = inputs.get("attention_mask")
            detox_labels = inputs.get("labels")
            
            detox_loss = torch.tensor(0.0, device=detox_input_ids.device)
            detox_outputs = None
            
            if detox_labels is not None:
                detox_outputs = model(input_ids=detox_input_ids, attention_mask=detox_attention_mask, labels=detox_labels)
                detox_loss = detox_outputs.loss
                detox_loss = torch.mean(detox_loss, axis=0)
    
            if return_outputs:
                return detox_loss, detox_outputs

            del detox_input_ids
            del detox_attention_mask
            del detox_labels

            del detox_outputs

            gc.collect()
            torch.cuda.empty_cache()
            
            return detox_loss
    
    def evaluate(self, *args, **kwargs):
        ds = self.eval_dataset
        ds = ds.select_columns(['detox_input_ids', 'detox_attention_mask', 'detox_labels'])
        ds = ds.rename_columns({'detox_input_ids': 'input_ids', 'detox_attention_mask': 'attention_mask', 'detox_labels': 'labels'})
        return super().evaluate(ds, ignore_keys=kwargs['ignore_keys'])


class TwoLosses:

    def __init__(self, model_path, output_dir, weight, top_n):
        self.model_path = model_path
        self.output_dir = output_dir
        self.weight = weight
        self.top_n = top_n
    
    def main(self):
    
        df_for_collator = prepare_dataset(self.top_n)
        train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        model = TwoLossesModel.from_pretrained(self.model_path)
        model.classification_head.weight.data.normal_(mean=0.0, std=0.02)
        if model.classification_head.bias is not None:
            model.classification_head.bias.data.zero_()
        
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.output_dir,
            eval_strategy="epoch",
            learning_rate=9e-5,
            optim='adafactor',
            lr_scheduler_type='cosine',
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=7,
            weight_decay=0.01,
            warmup_steps=50,
            logging_steps=30,
            save_total_limit=2,
            metric_for_best_model='loss',
            greater_is_better=False,
            remove_unused_columns=False,
            report_to="none",
            gradient_accumulation_steps=64,
            disable_tqdm=True,
            seed=42,
        )
        
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            weight=self.weight  # Pass the weight parameter
        )
        
        trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a TwoLossesModel with detoxification and classification tasks.')
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pre-trained model.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for saving checkpoints.')
    parser.add_argument('--weight', type=float, default=1., help='Weight for the classification loss.')
    parser.add_argument('--top_n', type=int, default=50, help='First N rows of dataset.')

    args = parser.parse_args()

    model_path = args.model_path
    output_dir = args.output_dir
    weight = args.weight
    top_n = args.top_n

    two_losses = TwoLosses(model_path, output_dir, weight, top_n)
    two_losses.main()