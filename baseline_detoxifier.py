#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset
import numpy as np
import pandas as pd
import huggingface_hub

token = 'Nope'
huggingface_hub.login(token)


# ## Baseline

# In[9]:


tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-large')

# Parameters
max_length = 128
batch_size = 4

dataset = load_dataset('s-nlp/synthdetoxm')

def preprocess_function(examples):
    inputs = []
    targets = []
    
    for tox, neu, lang in zip(examples['toxic_sentence'], examples['neutral_sentence'], examples['lang']):
        lang_map = {
            'de': 'german',
            'fr': 'french',
            'es': 'spanish',
            'ru': 'russian'
        }
        if tox:  # If toxic text is not empty
            inputs.append("Detoxify and return answer in " + lang_map[lang] + ": " + tox)
            targets.append(neu)
    
    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    
    # Tokenize target texts
    labels = tokenizer(targets, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").input_ids
    
    model_inputs["labels"] = labels
    
    return model_inputs

# Preprocess the dataset
train_test = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=66)
train_dataset = train_test['train']
val_dataset = train_test['test']

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load the base model
# model = AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-base')
model = AutoModelForSeq2SeqLM.from_pretrained('bigscience/mt0-large')

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results_mt0_large',
    num_train_epochs=4,
    learning_rate=9e-5,
    optim='adafactor',
    lr_scheduler_type='cosine',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    report_to="none",
    gradient_accumulation_steps=64,
)

# Initialize Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

def bullshit(preds):
    p1 = []
    for p in preds:
        p2 = []
        for token in p:
            p2.append(token if token != -100 else tokenizer.pad_token_id)
        p1.append(np.array(p2))
    p1 = torch.tensor(np.array(p1))
    return p1

# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)

# Generate predictions
predictions = trainer.predict(val_dataset)
decoded_preds = tokenizer.batch_decode(bullshit(predictions.predictions), skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(bullshit(predictions.label_ids), skip_special_tokens=True)

# Print some sample predictions and labels
for pred, label in zip(decoded_preds[:5], decoded_labels[:5]):
    print(f"Prediction: {pred}")
    print(f"Label: {label}")
    print("-" * 50)


# In[7]:


trainer.push_to_hub(repo_id='alexandro767/mT0-base-detoxifier-baseline-4L')


# ### Start from a checkpoint

# In[5]:


import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-large')

# Parameters
max_length = 128
batch_size = 4

# Load your dataset from Hugging Face
dataset = load_dataset('s-nlp/synthdetoxm')

# Function to preprocess the dataset
def preprocess_function(examples):
    # Initialize lists to store input and target texts
    inputs = []
    targets = []
    
    # Iterate over each batch of toxic and neutral texts
    for tox, neu, lang in zip(examples['toxic_sentence'], examples['neutral_sentence'], examples['lang']):
        lang_map = {
            'de': 'german',
            'fr': 'french',
            'es': 'spanish',
            'ru': 'russian'
        }
        if tox:  # If toxic text is not empty
            inputs.append("Detoxify and return answer in " + lang_map[lang] + ": " + tox)
            targets.append(neu)
    
    # Tokenize input texts
    model_inputs = tokenizer(inputs, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    
    # Tokenize target texts
    labels = tokenizer(targets, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt").input_ids
    
    model_inputs["labels"] = labels
    
    return model_inputs

# Preprocess the dataset
train_test = dataset['train'].train_test_split(test_size=0.2, shuffle=True, seed=66)
train_dataset = train_test['train']
val_dataset = train_test['test']

train_dataset = train_dataset.map(preprocess_function, batched=True)
val_dataset = val_dataset.map(preprocess_function, batched=True)

# Set format for PyTorch
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# Load the model from the checkpoint
checkpoint_dir = './results_mt0_large/checkpoint-100'  # Replace <step_number> with the actual checkpoint step number
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)

# Define training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='./results_mt0_large',
    num_train_epochs=4,
    learning_rate=9e-5,
    optim='adafactor',
    lr_scheduler_type='cosine',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="epoch",
    save_total_limit=2,
    predict_with_generate=True,
    report_to="none",
    gradient_accumulation_steps=64,
)

# Initialize Seq2SeqTrainer with the loaded model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    # compute_metrics=compute_metrics,  # Uncomment if you have a compute_metrics function
)

# # Evaluate the model
# eval_results = trainer.evaluate()
# print(eval_results)

# Generate predictions
predictions = trainer.predict(val_dataset)
decoded_preds = tokenizer.batch_decode(predictions.predictions, skip_special_tokens=True)
decoded_labels = tokenizer.batch_decode(predictions.label_ids, skip_special_tokens=True)

# # Print some sample predictions and labels
# for pred, label in zip(decoded_preds[:5], decoded_labels[:5]):
#     print(f"Prediction: {pred}")
#     print(f"Label: {label}")
#     print("-" * 50)


# ## For evaluation script

# In[8]:


def make_dataset(toxic, neutral, lang):
    assert len(toxic) == len(neutral) == len(lang)
    toxic_dataset = pd.DataFrame(data=np.array([toxic, neutral, lang]).T, columns=['toxic_sentence', 'neutral_sentence', 'lang'])
    neutral_dataset = pd.DataFrame(data=np.array([neutral, lang]).T, columns=['neutral_sentence', 'lang'])
    return toxic_dataset, neutral_dataset

toxic_dataset, neutral_dataset = make_dataset(val_dataset['toxic_sentence'], decoded_preds, val_dataset['lang'])
toxic_dataset.to_csv('val_toxic_3200_mt0_large.csv', index=None)
neutral_dataset.to_csv('val_neutral_3200_mt0_large.csv', index=None)

