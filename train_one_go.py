import re
import datetime,time
import pandas as pd
import string
import numpy as np
import wandb
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoTokenizer,AutoModelForSeq2SeqLM,
                      T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration,
                      AutoConfig, AutoModelForCausalLM,
                      get_linear_schedule_with_warmup)
from transformers.optimization import Adafactor, AdafactorSchedule

import random
import os
from collections import Counter

from sklearn.metrics import f1_score
import nltk
from nltk.tokenize import word_tokenize
import string
import evaluate
from transformers import GenerationConfig
import os
from accelerate import Accelerator
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel

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
  DATASET_TRAIN = ROOT + os.sep +'data/df_train_absa.csv'
  DATASET_TESTVAL = ROOT + os.sep +'data/df_testval.csv'
  
  MODEL_SAVE_PATH = ROOT + os.sep +'models/v1.pt'
  MODEL_PRETRAINED_ATE ='bigscience/mt0-xl'
  PRE_TRAINED_TOKENIZER_NAME ='bigscience/mt0-xl'

  MAX_SOURCE_LEN = 356
  MAX_TARGET_LEN = 40

  BATCH_SIZE_TRAIN_LBL = 4
  BATCH_SIZE_TEST = 2

  EPOCHS = 5
  LR = [3e-5, 1e-4, 2e-4, 3e-4]
  LR_IDX = 1
  EPS = 1e-5
  USE_LORA = False
  USE_CONSTANT_SCHEDULER = False
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q", "v"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM,
                        use_rslora=True
                        )


""" 
AIM: Train end to end ABSA 
"""
set_seed(config.SEED)
df_train = pd.read_csv(config.DATASET_TRAIN)
df_testval = pd.read_csv(config.DATASET_TESTVAL)

df_train.dropna(subset=['absa_input'],inplace=True)
df_testval.dropna(subset=['absa_input'],inplace=True)

punctuation = string.punctuation
punctuation = re.sub('-','',punctuation)
exact_match = evaluate.load("exact_match")

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

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

class ABSADataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'text': item['absa_input'],
            'target': item['absa_target'],
            'source_inputs_ids' : item['source_inputs_ids'].clone().detach().squeeze(),
            'source_attention_mask' : item['source_attention_mask'].clone().detach().squeeze(),
            'target_inputs_ids' : item['target_inputs_ids'].clone().detach().squeeze(),
            'target_attention_mask' : item['target_attention_mask'].clone().detach().squeeze()
        }

def get_dataloader(df_train, df_test_val, text_col, target_col):
  df_train['source_inputs_ids'], df_train['source_attention_mask'] = zip(* df_train.apply(lambda x: tokenize_function(x[text_col],tokenizer,config.MAX_SOURCE_LEN), axis=1))
  df_train['target_inputs_ids'], df_train['target_attention_mask'] = zip(* df_train.apply(lambda x: tokenize_function(x[target_col],tokenizer,config.MAX_TARGET_LEN), axis=1))
  df_test_val['source_inputs_ids'], df_test_val['source_attention_mask'] = zip(* df_test_val.apply(lambda x: tokenize_function(x[text_col],tokenizer,config.MAX_SOURCE_LEN), axis=1))
  df_test_val['target_inputs_ids'], df_test_val['target_attention_mask'] = zip(* df_test_val.apply(lambda x: tokenize_function(x[target_col],tokenizer,config.MAX_TARGET_LEN), axis=1))
  
  df_test, df_val = train_test_split(df_testval, test_size=0.50, random_state=config.SEED)

  train_dataset = ABSADataset(df_train)
  val_dataset = ABSADataset(df_val)
  test_dataset = ABSADataset(df_test)

  train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN_LBL, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE_TEST, shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE_TEST, shuffle=False)
  
  print('Train data size:', len(df_train))
  print('Test data size:', len(df_test))
  print('Val data size:', len(df_val))

  return train_dataloader, val_dataloader, test_dataloader


def load_model(base_name, accelerate):
    if 'mt5' in base_name:
      model = MT5ForConditionalGeneration.from_pretrained(base_name)
    else:
      model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    if accelerate==False:
        model = model.to(config.DEVICE)
    return model

def initialize_parameters(model, train_dataloader, optimizer_name, idx_lr):
  total_steps = len(train_dataloader) * config.EPOCHS
  if config.USE_CONSTANT_SCHEDULER==True:
    num_warmup_steps=0
  else:
    num_warmup_steps=1000
       
  if optimizer_name=='adam':
    optimizer = AdamW(model.parameters(), lr=config.LR[idx_lr], eps=config.EPS, correct_bias=False, no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)  

  elif optimizer_name=='ada':
    optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True, lr=None, clip_threshold=1.0)  
    scheduler = AdafactorSchedule(optimizer)

  autoconfig = AutoConfig.from_pretrained(config.MODEL_PRETRAINED_ATE)
  return optimizer, scheduler, autoconfig

def train_one_epoch(model, dataloader, optimizer, epoch, accelerate):
    total_t0 = time.time()
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.EPOCHS))
    print('Training...')

    train_losses = []
    model.train()

    for step, batch in enumerate(dataloader):

        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
        source_input_ids = batch['source_inputs_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_inputs_ids']
        target_attention_mask = batch['target_attention_mask']
        
        optimizer.zero_grad()
        
        if accelerate==False:
            source_input_ids = source_input_ids.to(config.DEVICE)
            source_attention_mask = source_attention_mask.to(config.DEVICE)
            target_input_ids = target_input_ids.to(config.DEVICE)
            target_attention_mask = target_attention_mask.to(config.DEVICE)
            
            outputs = model(input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        labels=target_input_ids, 
                        decoder_attention_mask=target_attention_mask)
        else:
            print('no valid option for accelerate')

        loss, _ = outputs[:2]
        loss_item = loss.item()
        print('Batch loss:', loss_item)
        train_losses.append(loss_item)

        if accelerate==False:
            loss.backward()
        else:
            print('no valid option for accelerate')

        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if config.USE_CONSTANT_SCHEDULER==False:
          scheduler.step()
          current_lr = scheduler.get_last_lr()[0]
        current_lr = optimizer.param_groups[-1]['lr']

    avg_train_loss = np.mean(train_losses)
    training_time = format_time(time.time() - total_t0)

    print("summary results")
    print("epoch | train loss | train time ")
    print(f"{epoch+1:5d} |   {avg_train_loss:.5f}  |   {training_time:}")
    return avg_train_loss, current_lr, model


def train_loop(lr_idx, model, tokenizer, train_dataloader, val_dataloader, optimizer, accelerate, gen_config):

  best_model=None
  best_val_loss = float('+inf')

  wandb.init(project="absa_ro_baseline_sem4_newdata_ONEGO", name="V12_mt0xl_whole_model",
             config={
                      "Description": "Train ABSA model with output pairs: <ATC1 is Pol1; ...;ATCn is PolN>",
                      "learning rate": config.LR[lr_idx],
                      "optimizer": "adam",
                      "constant scheduler": config.USE_CONSTANT_SCHEDULER,
                      "lora": config.USE_LORA,
                      "batch_size": config.BATCH_SIZE_TRAIN_LBL,
                      "MAX_SOURCE_LEN": config.MAX_SOURCE_LEN,
                      "MAX_TARGET_LEN": config.MAX_TARGET_LEN,
                      "epochs": config.EPOCHS,
                      "dataset train": f"{config.DATASET_TRAIN}",
                      "dataset test+val": f"{config.DATASET_TESTVAL}",
                      "pretrained model":config.MODEL_PRETRAINED_ATE,
                      "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
                      "model save path": config.MODEL_SAVE_PATH,
                      "Accelerate used": "False",
                      "Generate do_sample": gen_config.do_sample,
                      "Generate temperature": gen_config.temperature,
                      "Generate num_beams": gen_config.num_beams,
                      "Generate no_repeat_ngram_size": gen_config.no_repeat_ngram_size,
                    })

  for epoch in range(config.EPOCHS):
    train_loss, current_lr, model = train_one_epoch(model, train_dataloader, optimizer, epoch, accelerate)
    val_loss, val_em, val_f1  = eval(model, tokenizer, val_dataloader, epoch, accelerate)
    wandb.log({"Train Loss":train_loss, "Val Loss": val_loss, 
               "Val F1": val_f1, "Scheduler":current_lr,
               "Val exact match": val_em})

    if val_loss < best_val_loss:
      print(f"New best validation loss:{val_loss}")
      best_val_loss = val_loss
      best_model = model
      if accelerate==False:
        model.save_pretrained(config.MODEL_SAVE_PATH)
      else:
        print('No correct value for accelerate')
  
  print('TEST TIME')
  del model, train_dataloader, val_dataloader
  test_loss, test_em, test_f1 = eval(best_model,tokenizer, test_dataloader, 0, False)
                                                                        
  wandb.log({"Test Loss": test_loss, 
            "Test F1": test_f1,
            "Test exact match": test_em})

  print('Test F1:', test_f1)
  print('Test Exact Match:', test_em)
      

def f1(pred, target):
      return f1_score(target, pred, average='weighted')


def eval(model, tokenizer, dataloader, epoch, accelerate):
    total_t0 = time.time()
    print("Running Validation...")

    model.eval()

    valid_losses = []
    f1_scores = []
    em_scores = []

    for step, batch in enumerate(dataloader):
        source_input_ids = batch['source_inputs_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_inputs_ids']
        target_attention_mask = batch['target_attention_mask']
       
        with torch.no_grad():
            if accelerate==False:
                source_input_ids = source_input_ids.to(config.DEVICE)
                source_attention_mask= source_attention_mask.to(config.DEVICE)
                target_input_ids = target_input_ids.to(config.DEVICE)
                target_attention_mask = target_attention_mask.to(config.DEVICE)

                outputs = model(input_ids=source_input_ids,
                                attention_mask=source_attention_mask,
                                labels=target_input_ids,
                                decoder_attention_mask=target_attention_mask)
                generated_ids = model.generate(input_ids=source_input_ids,
                                    attention_mask=source_attention_mask,
                                    generation_config = gen_config
                                    )
            else:
                print('No valid option for accelerate')
                
            loss, _ = outputs[:2]
            valid_losses.append(loss.item())

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids[0]]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in target_input_ids]

            f1_val = f1(preds, target)
            em = exact_match.compute(predictions=preds, references=target)

            f1_scores.append(f1_val)
            em_scores.append(em['exact_match'])

            if step % 20 == 0 and not step == 0:
              for text,tar,pred in zip(batch['text'], target, preds):
                  print('Review:', text)
                  print('Target:',tar,'\nPrediction:',pred)
            else:
              print('Targets:',target,'\nPredictions:',preds)
            print('\nF1:',f1_val)
            print('Exact match:', em)

    avg_loss = np.mean(valid_losses)
    avg_f1 = np.mean(f1_scores)
    avg_em = np.mean(em_scores)
    training_time = format_time(time.time() - total_t0)
    print("\nsummary results")
    print("epoch | val loss | val acc | val f1 | val time")
    print(f"{epoch+1:5d} | {avg_loss:.5f} |  {avg_em:.5f} | {avg_f1:.5f} | {training_time:}")
    return avg_loss, avg_em, avg_f1

tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_TOKENIZER_NAME)
model = load_model(base_name=config.MODEL_PRETRAINED_ATE, accelerate=False)
if config.USE_LORA==True:
  model = prepare_model_for_kbit_training(model)
  model = get_peft_model(model, config.lora_config)
  model.print_trainable_parameters()

gen_config = GenerationConfig(
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.eos_token_id,
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=True,
    num_beams=5,
    temperature=0.2,
    num_return_sequences=1,
    no_repeat_ngram_size=4,
)


train_dataloader, val_dataloader, test_dataloader = get_dataloader(df_train, df_testval, 'absa_input', 'absa_target')

optimizer, scheduler, autoconfig = initialize_parameters(model, train_dataloader, 'adam', config.LR_IDX)

with torch.no_grad():
    torch.cuda.empty_cache()

train_loop(config.LR_IDX, model, tokenizer, 
          train_dataloader, val_dataloader,
          optimizer, False, gen_config) 
          
