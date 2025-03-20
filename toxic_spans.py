#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install sentencepiece -qqq')


# In[1]:


import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_dataset, concatenate_datasets, ClassLabel, Dataset
import numpy as np
import huggingface_hub
from sklearn.metrics import f1_score



# ## Вариант 2

# In[2]:


tokenizer = AutoTokenizer.from_pretrained('bigscience/mt0-large')


# In[3]:


def add_language_column(dataset, language):
    # lang_map = {'en': 0, 'de': 1, 'am': 2, 'es': 3, 'ru': 4, 'zh': 5, 'ar': 6, 'uk': 7, 'hi': 8}
    return dataset.add_column("Language", [language] * len(dataset))

dataset = load_dataset('textdetox/multilingual_toxic_spans')

datasets_with_language = []
for lang, data in dataset.items():
    dataset_with_language = add_language_column(data, lang)
    datasets_with_language.append(dataset_with_language)

combined_dataset = concatenate_datasets(datasets_with_language)

combined_dataset = combined_dataset.filter(lambda example: example['Negative Connotations'] is not None)


# In[4]:


def just_tokenize(examples):
    tokenized_input = tokenizer([examples['Sentence']], padding='max_length', max_length=128, return_tensors="pt", is_split_into_words=True)
    
    word_tokens = tokenized_input.tokens()
    word_ids = tokenized_input.word_ids()

    toxic_tokens = tokenizer([examples['Negative Connotations'].replace(',', ' ')], padding='max_length', max_length=128, return_tensors="pt", is_split_into_words=True).tokens()

    return {'Word tokens': word_tokens, 'Toxic tokens': toxic_tokens, 'input_ids': tokenized_input['input_ids'], 'attention_mask': tokenized_input['attention_mask']}

def align_labels_with_tokens(example):
    word_tokens = example['Word tokens']
    toxic_something = example['Toxic tokens']
    toxic_tokens = []
    for tk in toxic_something:
        if tk not in tokenizer.all_special_tokens:
            toxic_tokens.append(tk)
    
    token_classes = []
    for wt in word_tokens:
        if wt in toxic_tokens and wt!='▁' and wt!='' and wt!=' ':
            token_classes.append(1)
        else:
            token_classes.append(0)

    return {'labels': token_classes}


def go_squeeze(examples):
    input_ids = examples['input_ids'].squeeze()
    attention_mask = examples['attention_mask'].squeeze()
    labels = examples['labels'].squeeze()

    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}


# In[5]:


tokenized_dataset = combined_dataset.map(
    just_tokenize,
    batched=False,
)

tokenized_dataset = tokenized_dataset.map(
    align_labels_with_tokens,
    batched=False,
)


# In[6]:


model = AutoModelForTokenClassification.from_pretrained('bigscience/mt0-large')
# model.config.num_labels = 2


# In[7]:


train_test = tokenized_dataset.train_test_split(test_size=0.2, shuffle=True, seed=66)
train_dataset = train_test['train']
eval_dataset = train_test['test']

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[8]:


train_dataset = train_dataset.map(
    go_squeeze,
    batched=False,
)

eval_dataset = eval_dataset.map(
    go_squeeze,
    batched=False,
)


# In[10]:


batch_size = 2

args = TrainingArguments(
    output_dir="./mT0_large_for_token_classification",
    num_train_epochs=6,
    learning_rate=5e-5,
    optim='adafactor',
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=20,
    save_strategy="epoch",
    report_to="none",
    gradient_accumulation_steps=int(128/batch_size),
)


# In[11]:


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)
trainer.train()


# In[12]:


# Evaluate the model
eval_results = trainer.evaluate()
print(eval_results)


# In[13]:


def calculate_f1_score(true_labels: np.ndarray, predicted_labels: np.ndarray) -> float:
    """
    Calculate the F1-score for two 2D numpy arrays of true labels and predicted labels.

    Parameters
    ----------
    true_labels : np.ndarray
        A 2D numpy array of true labels (0 or 1).
    predicted_labels : np.ndarray
        A 2D numpy array of predicted labels (0 or 1).

    Returns
    -------
    float
        The F1-score.
    """
    # Ensure the input arrays have the same shape
    assert true_labels.shape == predicted_labels.shape, "The shape of true_labels and predicted_labels must be the same."

    # Flatten the arrays
    true_labels_flat = true_labels.flatten()
    predicted_labels_flat = predicted_labels.flatten()

    # Calculate the F1-score using sklearn's f1_score function
    f1 = f1_score(true_labels_flat, predicted_labels_flat)

    return f1


# In[14]:


# Generate predictions
train_pred = trainer.predict(train_dataset)
eval_pred = trainer.predict(eval_dataset)

train_argmax = np.argmax(train_pred.predictions, axis=2)
eval_argmax = np.argmax(eval_pred.predictions, axis=2)

f1_train = calculate_f1_score(train_dataset['labels'].numpy(), train_argmax)
f1_eval = calculate_f1_score(eval_dataset['labels'].numpy(), eval_argmax)

print('f1 train: ', f1_train)
print('f1 eval: ', f1_eval)


# In[90]:


trainer.push_to_hub(model_name ='alexandro767/mT0-large-toxic-spans-detection-preprocessed')


# In[93]:


for i, j in zip(train_test['test']['Word tokens'][115][:20], eval_argmax[115][:20]):
    print(i, j)


# In[78]:


example = 'нет друг я не оправдываюсь,.просто ты ёбаное мудло,я мать твою ебал, сестру твою ебал, кирпичами голову разбивал, с балкона сбрасывал переворачивал снова ебал, потом твоего отца в жопу ебал,потом завтавил твоего отца ебать в жопу твою сестру сука, и тебя пидараса заставлял всё это смотреть,а потом гавно жрать общее все что понасрали твои ебучие родные.'
tokenized_example = tokenizer([example], padding='max_length', max_length=128, return_tensors="pt", is_split_into_words=True)#.to('cuda:0')

dataset_ex = Dataset.from_dict(tokenized_example)

predictions_ex = trainer.predict(dataset_ex)
ex_argmax = np.argmax(predictions_ex.predictions, axis=2)

for i, j in zip(tokenized_example.tokens(), ex_argmax[0]):
    print(i, j)

