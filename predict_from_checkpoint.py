import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, MT5ForConditionalGeneration, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset, DatasetDict
import numpy as np
import huggingface_hub
from sklearn.metrics import f1_score
import gc
import torch.nn as nn
from tqdm import tqdm
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

token = 'your_token'
huggingface_hub.login(token)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

class TwoLossesModel(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.classification_head = nn.Linear(config.d_model, 3)  # Assuming binary classification

def prepare_dataset(top_n=None):
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


class PredictFromCheckpoint(checkpoint_path, model_name):

    self.checkpoint_path = checkpoint_path
    self.model_name = model_name

    
    def main():
    
        df_for_collator = prepare_dataset()
        train_test = df_for_collator.train_test_split(test_size=0.15, shuffle=True, seed=42)
        train_dataset = train_test['train']
        eval_dataset = train_test['test']

        model_cpt = TwoLossesModel.from_pretrained(self.checkpoint_path).to(device)
        tokenizer_cpt = AutoTokenizer.from_pretrained(self.model_name)
        
        eval_dataset.set_format(type='torch', columns=['detox_input_ids', 'detox_attention_mask', 'detox_labels'])
        
        eval_bs = 4
        detoxified_sentence = []
        
        for i in range(0, len(eval_dataset), eval_bs):
            batch = eval_dataset[i:i + eval_bs]
            with torch.no_grad():
                outputs = model_cpt.generate(
                    input_ids=batch['detox_input_ids'].to(device), 
                    attention_mask=batch['detox_attention_mask'].to(device), 
                    max_new_tokens=128
                )
            detoxified_outputs = tokenizer_cpt.batch_decode(outputs, max_lenght=128, skip_special_tokens=True)
            for do in detoxified_outputs:
                detoxified_sentence.append(do)
        
        toxic_sentence = eval_dataset['toxic_sentence']
        lang = eval_dataset['lang']
        
        return {'toxic_sentence': toxic_sentence,
                'detoxified_sentence': detoxified_sentence,
                'lang': lang}