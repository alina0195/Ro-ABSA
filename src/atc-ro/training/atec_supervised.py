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
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoTokenizer,AutoModelForSeq2SeqLM,
                      T5ForConditionalGeneration, T5Tokenizer, MT5ForConditionalGeneration,
                      AutoConfig, AutoModelForCausalLM,
                      get_linear_schedule_with_warmup)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from transformers.optimization import Adafactor, AdafactorSchedule
from transformers import GenerationConfig
from collections import Counter
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize

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
  DATASET_TRAIN = ROOT + os.sep +'data/df_train.csv'
  DATASET_TESTVAL = ROOT + os.sep +'data/df_testval.csv'
  
  MODEL_SAVE_PATH = ROOT + os.sep +'models/atec_sft.pt'
  MODEL_LARGE_PRETRAINED = 'bigscience/mt0-large'
  PRE_TRAINED_TOKENIZER_NAME ='bigscience/mt0-large'
 
  bleurt_checkpoint = 'BLEURT-20-D3'

  PROMPT = 'Categories of aspect terms: '  
  lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q", "v"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM
                        )
  
  MAX_SOURCE_LEN = 356
  MAX_TARGET_LEN = 30

  BATCH_SIZE_TRAIN_LBL = 4
  BATCH_SIZE_TEST = 2

  EPOCHS = 10
  LR = [5e-5, 1e-4, 2e-4, 3e-4, 1e-5]
  LR_IDX = 1
  EPS = 1e-5
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

""" 
AIM: Train a teacher model for Romanian ATE categories generations without ssl (just lbl plus synthetic data)
"""

set_seed(config.SEED)
df_train = pd.read_csv(config.DATASET_TRAIN)
df_testval = pd.read_csv(config.DATASET_TESTVAL)

df_train.dropna(subset=['text_cleaned'],inplace=True)
df_testval.dropna(subset=['text_cleaned'],inplace=True)


punctuation = string.punctuation
punctuation = re.sub('-','',punctuation)
scorer = evaluate.load("bleurt", module_type="metric", checkpoint=config.bleurt_checkpoint)
other_metrics = evaluate.combine([ "rouge","meteor","exact_match"])

def format_text_for_df(df, text_col_name):
    df[text_col_name] = df[text_col_name].apply(lambda x: x.lower())
    df[text_col_name] =  config.PROMPT + df[text_col_name] + ' </s>'
    return df
  
df_train = format_text_for_df(df_train, 'text_cleaned')
df_testval = format_text_for_df(df_testval, 'text_cleaned')

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
  return scores['scores']

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


class ATEDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
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
  # df_test.to_csv('test_newdata.csv', index=False)
  # df_val.to_csv('val_newdata.csv', index=False)

  train_dataset = ATEDataset(df_train)
  val_dataset = ATEDataset(df_val)
  test_dataset = ATEDataset(df_test)

  train_dataloader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE_TRAIN_LBL, shuffle=True)
  val_dataloader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE_TEST, shuffle=False)
  test_dataloader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE_TEST, shuffle=False)
  
  print('Train data size:', len(df_train))
  print('Test data size:', len(df_test))
  print('Val data size:', len(df_val))
  return train_dataloader, val_dataloader, test_dataloader

def load_model(base_name):
    if  'mt0' in base_name:
          model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    else:
          model = T5ForConditionalGeneration.from_pretrained(base_name)
    model = model.to(config.DEVICE)
    return model

tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_TOKENIZER_NAME)
model = load_model(base_name=config.MODEL_LARGE_PRETRAINED)
# model = prepare_model_for_kbit_training(model)
# model = get_peft_model(model, config.lora_config)
# model.print_trainable_parameters()
        
def initialize_parameters(model, train_dataloader, optimizer_name, idx_lr):
  total_steps = len(train_dataloader) * config.EPOCHS

  if optimizer_name=='adam':
    optimizer = AdamW(model.parameters(), lr=config.LR[idx_lr], eps=config.EPS, correct_bias=False, no_deprecation_warning=True)  # noqa: E501
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=1000, num_training_steps=total_steps)  # noqa: E501

  elif optimizer_name=='ada':
    optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True, lr=None, clip_threshold=1.0)  # noqa: E501
    scheduler = AdafactorSchedule(optimizer)

  autoconfig = AutoConfig.from_pretrained(config.MODEL_LARGE_PRETRAINED)
  return optimizer, scheduler, autoconfig

def train_one_epoch(model, dataloader, optimizer, epoch):
    total_t0 = time.time()
    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.EPOCHS))
    print('Training...')

    train_loss = 0
    model.train()

    for step, batch in enumerate(dataloader):

        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
        source_input_ids = batch['source_inputs_ids'].to(config.DEVICE)
        source_attention_mask = batch['source_attention_mask'].to(config.DEVICE)
        target_input_ids = batch['target_inputs_ids'].to(config.DEVICE)
        target_attention_mask = batch['target_attention_mask'].to(config.DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids=source_input_ids,
                    attention_mask=source_attention_mask,
                    labels=target_input_ids, # the forward function automatically creates the correct decoder_input_ids
                    decoder_attention_mask=target_attention_mask)

        loss, _ = outputs[:2]
        loss_item = loss.item()
        print('Batch loss:', loss_item)
        train_loss += loss_item

        loss.backward()
        
        optimizer.step()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # scheduler.step()
        # current_lr = scheduler.get_last_lr()[0]
        current_lr = optimizer.param_groups[-1]['lr']

    # calculate the average loss over all of the batches
    avg_train_loss = train_loss / len(dataloader)

    training_time = format_time(time.time() - total_t0)

    print("")
    print("summary results")
    print("epoch | train loss | train time ")
    print(f"{epoch+1:5d} |   {avg_train_loss:.5f}  |   {training_time:}")
    return avg_train_loss, current_lr, model

def train_loop(lr_idx, scorer, model, train_dataloader, val_dataloader, optimizer):

  best_valid_loss = float('+inf')
  train_losses = []
  val_losses = []
  vals_recall = []
  vals_bleurt = []
  vals_other_scores = []
  best_model=None
  

  wandb.init(project="atec_ro_baseline_sem4_newdata", name='mt0-large  lblData - v9 ct lr',
             config={
          "learning rate": config.LR[lr_idx],
          "optimizer": "adam",
          "epochs": config.EPOCHS,
          "prompt": config.PROMPT,
          "pretrained model":config.MODEL_LARGE_PRETRAINED,
          "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
          "model save path": config.MODEL_SAVE_PATH,
          })

  for epoch in range(config.EPOCHS):
    train_loss, current_lr, model = train_one_epoch(model, train_dataloader, optimizer, epoch)
    val_loss, val_bleurt, val_f1, val_recall, other_scores  = eval(model,val_dataloader,epoch,scorer)

    wandb.log({"Train Loss":train_loss, "Val Loss": val_loss, "Val BLEURT": val_bleurt,
               "Val F1": val_f1, "Val Recall":val_recall,
               "Val rouge1":other_scores['rouge1'],"Val rouge2":other_scores['rouge2'],
               "Val rougeL":other_scores['rougeL'],"Val rougeLsum":other_scores['rougeLsum'],
               "Val meteor":other_scores['meteor'],
               "Val exact match": other_scores['exact_match'], "Scheduler":current_lr})

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    vals_bleurt.append(val_bleurt)
    vals_recall.append(val_recall)
    vals_other_scores.append(other_scores)

    if val_loss <= best_valid_loss:
      print(f"New best loss:{val_loss}")
      best_valid_loss = val_loss
      best_model = model
      model.save_pretrained(config.MODEL_SAVE_PATH)
            
  return best_model

def f1(pred, target):
  return f1_score(target, pred, average='weighted')

def recall(pred, target):
  pred = [p.strip() for p in pred]
  target = [p.strip() for p in target]
  
  sum = 0
  already_seen = []
  for p in pred:
    if p in target and p not in already_seen:
      sum += 1
      already_seen.append(p)
  sum=sum/(len(target))
  return sum


gen_config = GenerationConfig(
    eos_token_id=model.config.eos_token_id,
    pad_token_id=model.config.eos_token_id,
    max_new_tokens = config.MAX_TARGET_LEN,
    return_dict_in_generate=True,
    output_scores=True,
    do_sample=True,
    num_beams=5,
    top_p=0.8,
    top_k=5,
    num_return_sequences=1,
    no_repeat_ngram_size=2,
)

def eval(model, dataloader, epoch, scorer):
    global tokenizer
    total_t0 = time.time()
    print("")
    print("Running Validation...")

    model.eval()

    valid_losses = []
    bleurt_scores = []
    f1_scores = []
    recalls = []
    other_scores = []

    for step, batch in enumerate(dataloader):
        source_input_ids = batch['source_inputs_ids'].to(config.DEVICE)
        source_attention_mask = batch['source_attention_mask'].to(config.DEVICE)
        target_input_ids = batch['target_inputs_ids'].to(config.DEVICE)
        target_attention_mask = batch['target_attention_mask'].to(config.DEVICE)
       
        with torch.no_grad():

            outputs = model(input_ids=source_input_ids,
                            attention_mask=source_attention_mask,
                            labels=target_input_ids,
                            decoder_attention_mask=target_attention_mask)
            generated_ids = model.generate(input_ids=source_input_ids,
                                attention_mask=source_attention_mask,
                                generation_config = gen_config
                                )
            
            loss, _ = outputs[:2]
            valid_losses.append(loss.item())

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids[0]]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in target_input_ids]
            preds_cleaned = [clean_predictions(p) for p in preds]
            target_cleaned = [clean_predictions(t) for t in target]

            recalls_current = [recall(preds_cleaned[idx],target_cleaned[idx]) for idx in range(0,len(target))]

            preds_cleaned_combined = [' or '.join(e) for e in preds_cleaned]
            targets_cleaned_combined = [' or '.join(t) for t in target_cleaned]

            bleurt = get_bleurt_score(scorer, preds_cleaned_combined, targets_cleaned_combined)
            f1_val = f1(preds_cleaned_combined, targets_cleaned_combined)
            other_metrics_val = other_metrics.compute(predictions=preds_cleaned_combined, references=targets_cleaned_combined)

            bleurt_scores += bleurt
            f1_scores.append(f1_val)
            recalls.extend(recalls_current)
            other_scores.append(other_metrics_val)

            print('Target Text:',target_cleaned,'\nGenerated Text:',preds_cleaned)
            print('F1:',f1_val)
            print('Recall:',recalls_current)
            print('BLEURT:',bleurt)
            print('Other metrics:',other_metrics_val['exact_match'],other_metrics_val['rougeL'])
            print('\n')
    avg_loss = np.mean(valid_losses)
    avg_bleurt = np.mean(np.abs(bleurt_scores))
    avg_f1 = np.mean(f1_scores)
    avg_recall = np.mean(recalls)
    training_time = format_time(time.time() - total_t0)
    other_scores=pd.DataFrame(other_scores)[['rouge1','rouge2','rougeL','rougeLsum','meteor','exact_match']]
    other_scores_mean = other_scores.mean()

    print("")
    print("summary results")
    print("epoch | val loss | val bleurt | val f1 | val recall | val time")
    print(f"{epoch+1:5d} | {avg_loss:.5f} |  {avg_bleurt:.5f} | {avg_f1:.5f} | {avg_recall:.5f} |{training_time:}")
    return avg_loss, avg_bleurt, avg_f1, avg_recall, other_scores_mean


train_dataloader, val_dataloader, test_dataloader = get_dataloader(df_train, df_testval, 'text_cleaned', 'all_categories')

optimizer, scheduler, autoconfig = initialize_parameters(model, 
                                                         train_dataloader, 
                                                         'adam', 
                                                         config.LR_IDX)


with torch.no_grad():
    torch.cuda.empty_cache()


""" RUN"""
model = model.to(config.DEVICE)
best_model = train_loop(config.LR_IDX, scorer, 
                        model, train_dataloader,
                        val_dataloader,optimizer)

print('TEST TIME')
test_loss, test_bleurt, test_f1, test_recall, test_other_scores = eval(best_model,
                                                                       test_dataloader, 
                                                                       0, scorer 
                                                                       )
wandb.log({"Test Loss": test_loss, "Test BLEURT": test_bleurt,
           "Test Recall":test_recall,"Test F1": test_f1,
           "Test rouge1":test_other_scores['rouge1'],
          "Test rouge2":test_other_scores['rouge2'],
          "Test rougeL":test_other_scores['rougeL'],
          "Test rougeLsum":test_other_scores['rougeLsum'],
          "Test meteor":test_other_scores['meteor'],
          "Test exact match": test_other_scores['exact_match']})

print('Test BLEURT:', test_bleurt)
print('Test F1:', test_f1)
print('Test Exact Match:', test_other_scores['exact_match'])
print('Test Rouge2:', test_other_scores['rouge2'])
print('Test Meteor:', test_other_scores['meteor'])
