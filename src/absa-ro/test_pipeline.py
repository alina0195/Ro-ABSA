
import re
import datetime,time
import pandas as pd
import string
import numpy as np
import wandb
import torch
import random
import os
import nltk
import string
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoTokenizer,AutoModelForSeq2SeqLM,
                      T5ForConditionalGeneration, T5Tokenizer, 
                      MT5ForConditionalGeneration,
                      AutoConfig, AutoModelForCausalLM,
                      get_linear_schedule_with_warmup)
from transformers.optimization import Adafactor, AdafactorSchedule
from peft import (LoraConfig, get_peft_model, 
                  prepare_model_for_kbit_training, 
                  TaskType, PeftModel)
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import GenerationConfig
from accelerate import Accelerator
nltk.download('punkt')
torch.set_warn_always(True)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

class config:
  SEED = 42
  ROOT = os.getcwd()
  DATASET = ROOT + os.sep +'data/df_testval.csv'
  
  MODEL_ALSA_ADAPTER = ROOT + os.sep +'models/alsa.pt'
  
  MODEL_ATE_ADAPTER= ROOT + os.sep + 'models/atec.pt'
  MODEL_ATE_BASE = 'bigscience/mt0-xl'
  MODEL_ALSA_BASE = 'bigscience/mt0-xl'
  
  TOKENIZER_ATE ='bigscience/mt0-xl'
  TOKENIZER_ALSA ='bigscience/mt0-xl'
  
  bleurt_checkpoint = 'BLEURT-20-D3'
  PROMPT_ALSA = 'Polarity opinion about {aspect} from the review: '
  PROMPT_ATE = 'Categories of opinion aspect terms: '  
  
  MAX_SOURCE_LEN = 512
  MAX_TARGET_LEN = 30

  BATCH_SIZE = 1
  lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q", "v"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        use_rslora=True
                        )
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


"""
Evaluate ABSA pipeline: review -> ATEC -> ALSA
Input: review
Output: ATC_1 is polarity; ATC_2 is polarity; ... ; ATC_n is polarity
"""

punctuation = string.punctuation
punctuation = re.sub('-','',punctuation)
exact_match = evaluate.load("exact_match")

set_seed(config.SEED)
def form_target_col_absa(row):
    categories = row['all_categories_old']
    polarities = row['all_polarities']
    
    categories = categories.split(';')
    polarities = polarities.split(';')
    
    pairs = []
    for cat, pol in zip(categories,polarities):
        pairs.append(cat.strip() + ' is ' + pol.strip().lower())
    return '; '.join(pairs)

def form_input_for_atec(df, input_col_name, output_col_name):
    df[output_col_name] = df[input_col_name].apply(lambda x: x.lower())
    df[output_col_name] =  config.PROMPT_ATE + df[output_col_name] + ' </s>'
    return df

def clean_predictions(pred):
    stop_words = ['the','i','a','or',')','un','o',' ']
    pred = pred.split(';')
    pred_new = []
    for x in pred:
      if x!= ' ':
        tokens = word_tokenize(x)
        tokens = [t for t in tokens if t.lower() not in stop_words]
        tokens = ' '.join(tokens)
        tokens = re.sub(r'\)',' ', tokens)
        tokens = re.sub(r'\(',' ', tokens)
        tokens = re.sub(' +', ' ', tokens)
        tokens = tokens.translate(str.maketrans('', '', punctuation))
        tokens = tokens.strip()
        if tokens != ' ':
          pred_new.append(tokens.strip())
    return pred_new

def remove_repeated_words(text):
  words = text.split()
  return " ".join(sorted(set(words), key=words.index))

def get_bleurt_score(scorer, preds, target):
  scores = scorer.compute(references=target, predictions=preds)
  avg_score = abs(np.array(scores['scores'])).mean()
  return scores['scores']

def f1(pred, target):
      return f1_score(target, pred, average='weighted')

def recall(preds, targets):
  targets = [t.strip() for t in targets]
  preds = [t.strip() for t in preds]
  
  sum = 0
  for p in preds:
    if p in targets:
      sum += 1
      print(f'Found {p}')
  sum=sum/(len(targets))
  return sum

def tokenize_function(text, tokenizer, max_len):
  encoded_dict = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True,
                padding='max_length',
                return_attention_mask=True,
                return_tensors='pt'
            )
  return encoded_dict['input_ids'], encoded_dict['attention_mask']

def tokenize_batch(tokenizer,batch, max_len):
  encoded_dict = tokenizer.batch_encode_plus(
              batch,
              add_special_tokens=True,
              max_length=max_len,
              truncation=True,
              padding='max_length',
              return_attention_mask=True,
              return_tensors='pt'
          )
  return encoded_dict['input_ids'], encoded_dict['attention_mask']


class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'text' : item['text_cleaned'],
            'input_atec': item['input_atec'],
            'target_absa': item['target_absa']
        }

def get_dataloader(df, shuffle, batch_size):
    ds = ABSADataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
    print('Data size:', len(df))
    return dl

def load_model(base_name,type, accelerate):
    if type=='t5':
      model = T5ForConditionalGeneration.from_pretrained(base_name)
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
   
    if accelerate==False:
        model = model.to(config.DEVICE)
    return model


df = pd.read_csv(config.DATASET)
df.dropna(subset=['text_cleaned'],inplace=True)
df, _ = train_test_split(df, test_size=0.50, random_state=config.SEED)

df['target_absa'] = df.apply(lambda x: form_target_col_absa(x), axis=1)
df = form_input_for_atec(df, 'text_cleaned', 'input_atec')
dl = get_dataloader(df=df, shuffle=False, batch_size=config.BATCH_SIZE)

model_atec = load_model(base_name=config.MODEL_ATE_BASE, type='mt0', accelerate=False)
model_atec = PeftModel.from_pretrained(model_atec, config.MODEL_ATE_ADAPTER)
model_atec = model_atec.merge_and_unload()
model_atec = prepare_model_for_kbit_training(model_atec)
model_atec = get_peft_model(model_atec, config.lora_config)

model_alsa = load_model(base_name=config.MODEL_ALSA_BASE, type='mt0', accelerate=False)
model_alsa = PeftModel.from_pretrained(model_alsa, config.MODEL_ALSA_ADAPTER)
model_alsa = model_alsa.merge_and_unload()
model_alsa = prepare_model_for_kbit_training(model_alsa)
model_alsa = get_peft_model(model_alsa, config.lora_config)

tokenizer_atec = AutoTokenizer.from_pretrained(config.TOKENIZER_ATE)
tokenizer_alsa = AutoTokenizer.from_pretrained(config.TOKENIZER_ALSA)

model_alsa = model_alsa.to(config.DEVICE)
model_atec = model_atec.to(config.DEVICE)
model_alsa.eval()
model_atec.eval()

gen_config_atec = GenerationConfig(
      eos_token_id=model_atec.config.eos_token_id,
      pad_token_id=model_atec.config.eos_token_id,
      max_new_tokens = config.MAX_TARGET_LEN,
      return_dict_in_generate=True,
      output_scores=True,
      do_sample=True,
      num_beams=20,
      temperature=0.7,
      num_return_sequences=1,
      no_repeat_ngram_size=2,
  )

gen_config_alsa  = GenerationConfig(
    eos_token_id=model_alsa.config.eos_token_id,
    pad_token_id=model_alsa.config.eos_token_id,
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    num_beams=3,
    do_sample=True,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
)

print("Running ...")
predictions = []
acc = 0
f1_vals = []
recall_vals_batch = []
recall_vals_pair = []

for step, batch in enumerate(dl):
  input_atec = batch['input_atec']
  target_absa = batch['target_absa']
  reviews = batch['text']
  
  print('New batch:')
  print('input atec:', input_atec)
  print('target absa:', target_absa)
  
  src_input_ids, src_att_mask = tokenize_batch(tokenizer_atec, input_atec, config.MAX_SOURCE_LEN)
  src_input_ids = src_input_ids.to(config.DEVICE)
  src_att_mask = src_att_mask.to(config.DEVICE)
  
  # extract ATEC
  with torch.no_grad():
    generated_ids = model_atec.generate(
                input_ids = src_input_ids,
                attention_mask = src_att_mask,
                generation_config = gen_config_atec)
   
    categories = [tokenizer_atec.decode(tok, skip_special_tokens=True, clean_up_tokenization_space=True) for tok in generated_ids[0]]
    categories_cleaned = [clean_predictions(p) for p in categories]
    print('Categories predicted cleaned:')
    print(categories_cleaned)
    
    reviews_absa_batch = []
    for idx, categories_batch in enumerate(categories_cleaned):
      review_absa_pairs = []
      for category in categories_batch:
        absa_input = config.PROMPT_ALSA.format(aspect=category) + reviews[idx] +' </s>'
        src_input_ids, src_att_mask = tokenize_function(absa_input, tokenizer_alsa, config.MAX_SOURCE_LEN)
        src_input_ids = src_input_ids.to(config.DEVICE)
        src_att_mask = src_att_mask.to(config.DEVICE)
        with torch.no_grad():
          generated_ids = model_alsa.generate(
                input_ids = src_input_ids,
                attention_mask = src_att_mask,
                generation_config = gen_config_alsa)
          
        polarity = [tokenizer_alsa.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids[0]]
        
        absa_output = category.strip() + ' is ' + polarity[0].strip()
        review_absa_pairs.append(absa_output)
      
      review_absa = '; '.join(review_absa_pairs)
      current_recall_pairs = recall(preds=review_absa.split(';'), targets = target_absa[idx].split(';'))
      recall_vals_pair.append(current_recall_pairs)
      reviews_absa_batch.append(review_absa)
    
    print('ABSA Batch target:')
    print(target_absa)
    print('ABSA Batch predictions:')
    print(reviews_absa_batch)
    
    current_f1_batch = f1(pred=reviews_absa_batch, target=target_absa)
    print('F1 on batch level:', current_f1_batch)
    
    current_recall_batch = recall(preds=reviews_absa_batch, targets=target_absa)
    print('Recall on batch level:', current_recall_batch)
    
    f1_vals.append(current_f1_batch)
    recall_vals_batch.append(current_recall_batch)
    
    print('*'*100)

wandb.init(project="pipeline_absa", name="V2",
             config={
               'model_atc_adapter': config.MODEL_ATE_ADAPTER,
               'model_atc_base': config.MODEL_ATE_BASE,
               'model_aLSC_adapter': config.MODEL_ALSA_ADAPTER,
               'model_aLSC_base': config.MODEL_ALSA_BASE,
               'alsc adapter f1': '0.90',
               'alsc adapter exact match': '0.91',
               'atc adapter f1': '0.59',
               'atc adapter detail': 'sft',
               'gen_config_alsc':'num_beams=3,no_reapeat_ngram=2',
               'gen_config_atc':'do_sample=True, num_beams=20, temp=0.2, no_repeat_ngram=2'
             }) 

wandb.log({
  'F1 batch level': np.mean(f1_vals),
  'Recall batch level': np.mean(recall_vals_batch),
  'Recall pair level': np.mean(recall_vals_pair),
})
print('Final recall pair level:', np.mean(recall_vals_pair))
print('Final recall batch level:', np.mean(recall_vals_batch))
print('Final f1 batch level:', np.mean(f1_vals))