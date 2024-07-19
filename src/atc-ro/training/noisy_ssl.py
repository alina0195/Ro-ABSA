# Classic self training where the initially trained (teacher) learns iterratively a different better model (student) to solve the task
 
import re
import datetime,time
import pandas as pd
import numpy as np
import os
import nltk
import random
import wandb
import torch
import string
import evaluate
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (AdamW, AutoTokenizer,AutoModelForSeq2SeqLM,
                      T5ForConditionalGeneration, MT5ForConditionalGeneration,
                      GenerationConfig,
                      get_linear_schedule_with_warmup)

from transformers.optimization import Adafactor, AdafactorSchedule
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PeftModel
from nltk.tokenize import word_tokenize

nltk.download('punkt')
torch.set_warn_always(True)
os.environ['TOKENIZERS_PARALLELISM'] = "false"


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
  DATASET_ULBL = ROOT + os.sep +'data/unlabeled_reviews.csv'
  
  MODEL_SAVE_PATH = ROOT + os.sep +'models/atec_noisy_ssl.pt'
  
  MODEL_TEACHER_ADAPTER = ROOT + os.sep + 'models/atec.pt'
  MODEL_TEACHER_BASE = 'bigscience/mt0-xl'

  MODEL_STUDENT = 'bigscience/mt0-xl'
  PRE_TRAINED_TOKENIZER_NAME ='bigscience/mt0-base'

  DF_TRAIN_SAVE_PATH =ROOT + os.sep + 'data/df_train_plus_pslbls.csv'
  DF_ULBL_SAVE_PATH = ROOT + os.sep +'data/df_ulbl_reviews_left.csv'
  
  PROMPT = 'Categories of opinion aspect terms: '  

  MAX_SOURCE_LEN = 356
  MAX_TARGET_LEN = 20

  BATCH_SIZE_TRAIN_LBL = 2
  BATCH_SIZE_ULBL = 2
  BATCH_SIZE_TEST = 2
  
  accumulation_steps = 4
  lora_config = LoraConfig(
                        r=16,
                        lora_alpha=32,
                        target_modules=["q", "v"],
                        lora_dropout=0.05,
                        bias="none",
                        task_type=TaskType.SEQ_2_SEQ_LM
                        )
  EPOCHS = 6
  MAX_ITER = 6
  LR = [1e-4, 2e-4, 3e-4]
  LR_IDX = 0
  EPS = 1e-5
  DROPOUT = 0.2
  THRESHOLDS = [0.65, 0.80, 0.85, 0.90, 0.95]
  DATA_ORIGIN_TYPE = ['unlabeled','manual']
  DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
  BLACKLIST = ["price; quality",'product', 'quality; product', 'quality; price', 
               "product; quality; price", "product; price; quality",
               "product; quality", "price; product; quality", 
               "quality; price; product", "quality; product; price",
               "price; product", "price; quality", "product; quality; product",
               "price; quality; product", "product; price",
               "quality; price; product", "shop diversity"]

set_seed(config.SEED)

df_train = pd.read_csv(config.DATASET_TRAIN)
df_testval = pd.read_csv(config.DATASET_TESTVAL)
df_ulbl =  pd.read_csv(config.DATASET_ULBL)

df_train = df_train[~(df_train['data_origin']=='bt_1chain')]
df_train = df_train[~(df_train['data_origin']=='bt_2chain')]

df_train.dropna(subset=['text_cleaned'], inplace=True)
df_testval.dropna(subset=['text_cleaned'], inplace=True)

df_ulbl['data_origin'] =  config.DATA_ORIGIN_TYPE[0]
df_ulbl['all_categories'] = ''
df_ulbl['id'] = [i for i in range(0,len(df_ulbl))]

punctuation = string.punctuation
punctuation = re.sub('-','',punctuation)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def format_text_for_df(df, text_col_name):
    df[text_col_name] = df[text_col_name].apply(lambda x: x.lower())
    df[text_col_name] =  config.PROMPT + df[text_col_name] + ' </s>'
    return df

df_train = format_text_for_df(df_train, 'text_cleaned')
df_ulbl = format_text_for_df(df_ulbl, 'text_cleaned')
df_testval = format_text_for_df(df_testval, 'text_cleaned')
df_train['has_bt']='False'

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


def get_lbl_dataloader(df, text_col, target_col, tokenizer, shuffle, batch_size):
    df['source_inputs_ids'], df['source_attention_mask'] = zip(* df.apply(lambda x: tokenize_function(x[text_col],
                                                                                                      tokenizer,
                                                                                                      config.MAX_SOURCE_LEN), 
                                                                          axis=1))
    df['target_inputs_ids'], df['target_attention_mask'] = zip(* df.apply(lambda x: tokenize_function(x[target_col],
                                                                                                      tokenizer,
                                                                                                      config.MAX_TARGET_LEN), 
                                                                          axis=1))
    ds = ATEDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    print('LBL data size:', len(df))
    return dl


def get_lbl_dataloaders(df_train, df_test, df_val, text_col, target_col, tokenizer):
   
  train_dataloader = get_lbl_dataloader(df_train, text_col, target_col, tokenizer, True, config.BATCH_SIZE_TRAIN_LBL)
  test_dataloader = get_lbl_dataloader(df_test, text_col, target_col, tokenizer, False,config.BATCH_SIZE_TEST )
  val_dataloader = get_lbl_dataloader(df_val, text_col, target_col, tokenizer,False, config.BATCH_SIZE_TEST)

  return train_dataloader, val_dataloader, test_dataloader


def load_model(base_name, accelerate):
    if 'mt0' in base_name.lower():
      model = AutoModelForSeq2SeqLM.from_pretrained(base_name)
    elif 'mt5' in base_name.lower():
      model = MT5ForConditionalGeneration.from_pretrained(base_name)
    else:
      model = T5ForConditionalGeneration.from_pretrained(base_name)
    if accelerate==False:
        model = model.to(config.DEVICE)
    return model

def initialize_parameters(model, train_dataloader, optimizer_name, idx_lr):
  total_steps = len(train_dataloader) * config.EPOCHS

  if optimizer_name=='adam':
    optimizer = AdamW(model.parameters(), lr=config.LR[idx_lr], eps=config.EPS, correct_bias=False, no_deprecation_warning=True)  # noqa: E501
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)  # noqa: E501

  elif optimizer_name=='ada':
    optimizer = Adafactor(model.parameters(), relative_step=True, warmup_init=True, lr=None, clip_threshold=1.0)  # noqa: E501
    scheduler = AdafactorSchedule(optimizer)

  return optimizer, scheduler


def train_one_epoch(model, tokenizer, dataloader, optimizer, epoch, accelerate):
    total_t0 = time.time()
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, config.EPOCHS))
    train_loss = 0
    model.train()

    accumulation_counter = 0

    for step, batch in enumerate(dataloader):

        if step % 200 == 0 and step != 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
        source_input_ids = batch['source_inputs_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_inputs_ids']
        target_attention_mask = batch['target_attention_mask']
        
        if accelerate==False:
            source_input_ids = source_input_ids.to(config.DEVICE)
            source_attention_mask = source_attention_mask.to(config.DEVICE)
            target_input_ids=target_input_ids.to(config.DEVICE)
            target_attention_mask=target_attention_mask.to(config.DEVICE)
            
            outputs = model(input_ids=source_input_ids,
                        attention_mask=source_attention_mask,
                        labels=target_input_ids, # the forward function automatically creates the correct decoder_input_ids
                        decoder_attention_mask=target_attention_mask)

        loss, prediction_scores = outputs[:2]
        loss_item = loss.item()
        train_loss += loss_item
        
        # Scale the loss by the number of accumulation steps
        loss = loss / config.accumulation_steps
        if accelerate==False:
            loss.backward()
        accumulation_counter += 1
        
        # Perform the optimization step only after the specified number of accumulation steps
        if accumulation_counter % config.accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # scheduler.step()
        current_lr = optimizer.param_groups[-1]['lr']

    # calculate the average loss over all of the batches
    avg_train_loss = train_loss / len(dataloader)

    training_time = format_time(time.time() - total_t0)

    print("")
    print("summary results")
    print("epoch | train loss | train time ")
    print(f"{epoch+1:5d} |   {avg_train_loss:.5f}  |   {training_time:}")
    return avg_train_loss, current_lr, model


def train_loop(iter, model, lr_idx,other_metrics, tokenizer, train_dataloader, val_dataloader, optimizer, gen_config, accelerate):
  
  best_valid_loss = float('+inf')
  best_model=None

  for epoch in range(config.EPOCHS):
    train_loss, current_lr, model = train_one_epoch(model, tokenizer, train_dataloader, 
                                             optimizer, epoch, accelerate)
    val_loss, val_f1, val_recall, other_scores  = eval(model, tokenizer, val_dataloader,
                                                                   epoch, other_metrics,gen_config, accelerate)

    wandb.log({"Train Loss":train_loss, "Val Loss": val_loss,
               "Val F1": val_f1, "Val Recall":val_recall,
               "Val rouge1":other_scores['rouge1'],"Val rouge2":other_scores['rouge2'],
               "Val rougeL":other_scores['rougeL'],"Val rougeLsum":other_scores['rougeLsum'],
               "Val meteor":other_scores['meteor'],
               "Val exact match": other_scores['exact_match'], "Scheduler":current_lr})

    if val_loss <= best_valid_loss:
      print(f"New best val loss:{val_loss}")
      best_valid_loss = val_loss
      best_model = model
      
  del model
  return best_model


def eval(model, tokenizer, dataloader, epoch, other_metrics,gen_config, accelerate):
    total_t0 = time.time()
    print("")
    print("Running Validation...")

    model.eval()
        
    valid_losses = []
    f1_scores = []
    recalls = []
    other_scores = []

    for step, batch in enumerate(dataloader):
        source_input_ids = batch['source_inputs_ids']
        source_attention_mask = batch['source_attention_mask']
        target_input_ids = batch['target_inputs_ids']
        target_attention_mask = batch['target_attention_mask']
       
        with torch.no_grad():
            if accelerate==False:
                source_input_ids = source_input_ids.to(config.DEVICE)
                source_attention_mask = source_attention_mask.to(config.DEVICE)
                target_input_ids = target_input_ids.to(config.DEVICE)
                target_attention_mask =  target_attention_mask.to(config.DEVICE)

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
            
            f1_val = f1(preds_cleaned_combined, targets_cleaned_combined)
            other_metrics_val = other_metrics.compute(predictions=preds_cleaned_combined, references=targets_cleaned_combined)

            f1_scores.append(f1_val)
            recalls.extend(recalls_current)
            other_scores.append(other_metrics_val)

            if step % 10 == 0 and not step == 0:
                reviews = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in source_input_ids]
                print('Reviews:',reviews)
            print('Target Text:',target,'\nGenerated Text:',preds)
            print('F1:',f1_val)
            print('Recall:',recalls_current)
            print('Other metrics:',other_metrics_val['exact_match'],other_metrics_val['rougeL'])
            print('')
    avg_loss = np.mean(valid_losses)
    avg_f1 = np.mean(f1_scores)
    avg_recall = np.mean(recalls)
    training_time = format_time(time.time() - total_t0)
    other_scores=pd.DataFrame(other_scores)[['rouge1','rouge2','rougeL','rougeLsum','meteor','exact_match']]
    other_scores_mean = other_scores.mean()

    print("")
    print("summary results")
    print("epoch | val loss | val f1 | val recall | val time")
    print(f"{epoch+1:5d} | {avg_loss:.5f} | {avg_f1:.5f} | {avg_recall:.5f} |{training_time:}")
    return avg_loss, avg_f1, avg_recall, other_scores_mean


def generate_labels(model, tokenizer, df_ulbl, thresh_idx,gen_config, accelerate):
  print('Generate pseudo-labels...')

  scores = []
  pslabels = []
  notcommons = []
  model.eval() # de-activate model noise (dropout)
  
  for _, row in df_ulbl.iterrows():
    source_input_ids, attention_mask = tokenize_function(row['text_cleaned'], tokenizer, config.MAX_SOURCE_LEN)
    source_input_ids = source_input_ids.to(config.DEVICE)
    attention_mask = attention_mask.to(config.DEVICE)
    with torch.no_grad():
        generated_ids = model.generate(
                input_ids = source_input_ids,
                attention_mask = attention_mask,
                generation_config = gen_config
                )

    score = generated_ids.sequences_scores[0]
    score = np.exp(score.cpu().numpy())
    
    pslabel = [tokenizer.decode(tok, skip_special_tokens=True, clean_up_tokenization_space=True) for tok in generated_ids[0]]
    
    pslabel_joined = ''.join(pslabel)   
    
    if pslabel_joined.strip() not in config.BLACKLIST:
        is_not_common = 'True'
    else:
        is_not_common = 'False'
    notcommons.append(is_not_common)
    
    wandb.log({"Current Threshold": config.THRESHOLDS[thresh_idx] })
    if score > config.THRESHOLDS[thresh_idx] and is_not_common=='True':
        print(row['text_cleaned'])
        print(f'\n - score = {score} : {pslabel_joined}\n')

    pslabels.append(pslabel_joined)
    scores.append(score)
    
  new_df_ulbl = df_ulbl.copy()
  new_df_ulbl['score']=scores
  new_df_ulbl['all_categories']=pslabels
  new_df_ulbl['notcommons']=notcommons
  
  new_df_ulbl = new_df_ulbl[new_df_ulbl['score'] > config.THRESHOLDS[thresh_idx]]
  new_df_ulbl = new_df_ulbl[new_df_ulbl['notcommons']=='True']
  
  return new_df_ulbl


def test_model(model, tokenizer, test_dataloader,other_metrics,gen_config, accelerate):
    print('TEST TIME')
    test_loss, test_f1, test_recall, test_other_scores = eval(model,
                                                                tokenizer,
                                                            test_dataloader, 
                                                            0, other_metrics,
                                                            gen_config,accelerate)
    wandb.log({"Test Loss": test_loss,
            "Test Recall":test_recall,"Test F1": test_f1,
            "Test rouge1":test_other_scores['rouge1'],
            "Test rouge2":test_other_scores['rouge2'],
            "Test rougeL":test_other_scores['rougeL'],
            "Test rougeLsum":test_other_scores['rougeLsum'],
            "Test meteor":test_other_scores['meteor'],
            "Test exact match": test_other_scores['exact_match']})

    print('Test F1:', test_f1)
    print('Test Exact Match:', test_other_scores['exact_match'])
    print('Test Rouge2:', test_other_scores['rouge2'])
    print('Test Meteor:', test_other_scores['meteor'])
  
    return test_loss, test_other_scores['exact_match']


def update_train_data(df_train, df_high_probs, df_ulbl):
    df_ulbl_copy = df_ulbl.copy()
    
    print('Updating train data...')
    df_high_probs.drop(columns=['score','notcommons'], inplace=True)
    
    df_ulbl_copy = df_ulbl_copy.set_index('text_cleaned')
    df_high_probs = df_high_probs.set_index('text_cleaned')

    df_ulbl_copy = df_ulbl_copy.drop(index=df_high_probs.index)
    
    df_high_probs.reset_index(inplace=True)
    df_ulbl_copy.reset_index(inplace=True)
    
    df_high_probs['has_bt']='False'
    df_high_probs['data_origin']=config.DATA_ORIGIN_TYPE[0]
    
    df_train_new = pd.concat([df_train, df_high_probs], axis=0, ignore_index=True)
    
    return df_train_new, df_ulbl_copy


class BTDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        return {
            'text': item['text_cleaned'],
        }


def translate_batch(model, text_list, tokenizer_src, src_lang, tgt_lang):
    tokenizer_src.src_lang = src_lang
    inputs = tokenizer_src(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_length=512,
                                          forced_bos_token_id=tokenizer_src.lang_code_to_id[tgt_lang],
                                          num_beams=5, do_sample=True, early_stopping=True)
    decoded_tokens = [tokenizer_src.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_tokens]
    return [d for d in decoded_tokens]


def backtranslation_batch(model,text_list, tokenizer_src, tokenizer_tgt, tokenizer_tgt2, src_lang, tgt_lang, tgt_lang2=None):
    translated_text = translate_batch(model,text_list, tokenizer_src, src_lang, tgt_lang)
    if tgt_lang2==None:
        text_bt = translate_batch(model, translated_text,tokenizer_tgt, tgt_lang, src_lang)
    else:
        translated_text = translate_batch(model, translated_text, tokenizer_tgt, tgt_lang, tgt_lang2)
        text_bt = translate_batch(model, translated_text, tokenizer_tgt2, tgt_lang2, src_lang)
    return text_bt


def run_backtranslation(df):
    print("Running Backtranslation...")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    model=model.to(config.DEVICE)
    
    df_target = df[df['has_bt']=='False']
    df_target = df_target.sample(frac=0.1)
    
    if len(df_target)>0:
        print('Current target:', df_target[['id','has_bt','data_origin', 'all_categories']])
        
        # clean prompt and end token 
        df_target['text_cleaned'] = df_target['text_cleaned'].str.replace(config.PROMPT, '')
        df_target['text_cleaned'] = df_target['text_cleaned'].str.replace('</s>', '')
        
        ds = BTDataset(df_target)
        dl = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
        
        model.eval()
        bt_instances = []
        for batch in dl:
            text_list = batch['text']
            text_list_bt = backtranslation_batch( model = model,
                                        text_list=text_list, 
                                        tokenizer_src=tokenizer,
                                        tokenizer_tgt=tokenizer,
                                        tokenizer_tgt2 = None,
                                        src_lang='ron_Latn',
                                        tgt_lang='eng_Latn',
                                        tgt_lang2=None
                                        )
            bt_instances.extend(text_list_bt)
            
        df_target['bt'] = bt_instances
        
        df_target = format_text_for_df(df_target, 'bt')
        
        df_target.loc[df_target['data_origin'] == config.DATA_ORIGIN_TYPE[0], 'data_origin'] = config.DATA_ORIGIN_TYPE[0] + '_bt_EnRo_noisyStud'
        df_target.loc[df_target['data_origin'] == config.DATA_ORIGIN_TYPE[1], 'data_origin'] = config.DATA_ORIGIN_TYPE[1] + '_bt_EnRo_noisyStud'
        df_target['has_bt'] = 'True'
        
        df_new = df_target[['id','bt','all_categories','all_ate','all_polarities','data_origin','has_bt']].copy()
        df_new.rename(columns={'bt':'text_cleaned'}, inplace=True)
        df_new['has_bt']='True'
        df.loc[df['text_cleaned'].isin(df_target['text_cleaned']),'has_bt']='True'
        
        print('Augmenting train data...')
        df_aug = pd.concat([df, df_new],axis=0, ignore_index=True)
        
        if 'text_cleaned' not in df_aug.columns:
            print('ERRRRROOOOR')
            return df
        else:
            df_aug_unique = df_aug.drop_duplicates(subset=['text_cleaned'])
            return df_aug_unique
    else:
        return df


def set_dropout(model, dropout_rate):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = dropout_rate


def self_training_noisy_student(df_train, df_testval, df_ulbl, max_iter):
  other_metrics = evaluate.combine([ "rouge","meteor","exact_match"])
  tokenizer = AutoTokenizer.from_pretrained(config.PRE_TRAINED_TOKENIZER_NAME)
  

  df_test, df_val = train_test_split(df_testval, test_size=0.50, random_state=config.SEED)

  test_dataloader = get_lbl_dataloader(df = df_test, text_col='text_cleaned', target_col= 'all_categories',
                                       tokenizer=tokenizer, shuffle=False, batch_size=config.BATCH_SIZE_TEST)
  val_dataloader = get_lbl_dataloader(df = df_val, text_col='text_cleaned', target_col= 'all_categories',
                                       tokenizer=tokenizer, shuffle=False, batch_size=config.BATCH_SIZE_TEST)
  best_test_em = float('-inf')
  df_ulbl_high_probs = [1]
  
  
  wandb.init(project="atec_ssl_noisy", name=f'V1',
        config={
          "learning rate": config.LR[config.LR_IDX],
          "optimizer": "adam",
          "epochs": config.EPOCHS,
          "prompt": config.PROMPT,
          "pretrained model for STUDENT":config.MODEL_STUDENT,
          "pretrained model for TEACHER":config.MODEL_TEACHER_ADAPTER,
          "pretrained tokenizer":config.PRE_TRAINED_TOKENIZER_NAME,
          "model save path": config.MODEL_SAVE_PATH,
          "Threshold": config.THRESHOLDS,
          "Batch size - TRAIN": config.BATCH_SIZE_TRAIN_LBL,
          "Blacklist": config.BLACKLIST,
          "new train dataset path": config.DF_TRAIN_SAVE_PATH,
          "left unlabeled dataset path": config.DF_ULBL_SAVE_PATH,
          })
  
  """Algorithm"""
  # 1. learn a teacher model on labeled data
  teacher_model = load_model(base_name=config.MODEL_TEACHER_BASE, accelerate=False)
  teacher_model= PeftModel.from_pretrained(teacher_model, config.MODEL_TEACHER_ADAPTER)
  teacher_model= teacher_model.merge_and_unload()
  teacher_model = prepare_model_for_kbit_training(teacher_model)
  teacher_model = get_peft_model(teacher_model, config.lora_config)
  
  gen_config = GenerationConfig(
        eos_token_id=teacher_model.config.eos_token_id,
        pad_token_id=teacher_model.config.eos_token_id,
        max_new_tokens = config.MAX_TARGET_LEN,
        return_dict_in_generate=True,
        output_scores=True,
        do_sample=True,
        num_beams=20,
        temperature=0.75,
        num_return_sequences=1,
        no_repeat_ngram_size=4,
    )
  
  for iter in range(max_iter):
        if iter >= len(config.THRESHOLDS):
            thresh_idx = len(config.THRESHOLDS)-1
        else:
            thresh_idx = iter
            
        # 2. generate soft/hard pseudo-labeled for clean unlabeled data
        print('Unlabeled data size:', len(df_ulbl))
        print('Current train data size:', len(df_train))
        df_ulbl_high_probs  =  generate_labels(teacher_model, tokenizer, df_ulbl, thresh_idx, gen_config, False)
        print(f'#New pslabels generated: {len(df_ulbl_high_probs)}')
        
        
        df_train, df_ulbl = update_train_data(df_train, df_ulbl_high_probs, df_ulbl)
        wandb.log({"#Ulbl instances remaining":len(df_ulbl)})
        df_ulbl[['id','text_cleaned','data_origin']].to_csv(config.DF_ULBL_SAVE_PATH, index=False)
                
        # prepare noisy data
        df_train = run_backtranslation(df_train)
        wandb.log({"#Total train instances after injecting noise":len(df_train)})
        df_train[['id','text_cleaned','all_categories','all_ate','all_polarities','data_origin']].to_csv(config.DF_TRAIN_SAVE_PATH, index=False)
        
        print('TRAIN DATAFRAME SIZE AFTER AUGMENTATION:')
        print(len(df_train))
        
        train_dataloader = get_lbl_dataloader(df = df_train, 
                                              text_col='text_cleaned', 
                                              target_col= 'all_categories',
                                              tokenizer=tokenizer, shuffle=True, 
                                              batch_size=config.BATCH_SIZE_TRAIN_LBL)
       
        # 3. learn a larger student model which minimizes cross entropy loss on lbl and ulbl data
        teacher_model = None
        student_model = load_model(base_name=config.MODEL_STUDENT, accelerate=False)
        student_model = prepare_model_for_kbit_training(student_model)
        student_model = get_peft_model(student_model, config.lora_config)
        student_model.print_trainable_parameters()
        set_dropout(student_model, config.DROPOUT) # the dropout is deactivated during generation if we use model.eval()
        
        optimizer, scheduler = initialize_parameters(student_model, train_dataloader, 
                                                     'adam', config.LR_IDX)
        student_model = train_loop(iter, student_model, config.LR_IDX,  
                                    other_metrics, 
                                    tokenizer, train_dataloader,
                                    val_dataloader,optimizer,gen_config, False)
        _, test_exact_match = test_model(student_model, tokenizer,
                                                 test_dataloader, 
                                                 other_metrics,gen_config, False)
        if best_test_em <= test_exact_match:
            best_test_em = test_exact_match
            student_model.save_pretrained(config.MODEL_SAVE_PATH)
            
        if len(df_ulbl_high_probs)<=1:
              print('No new labels generated.\Exiting the loop...')
              break

        # 4. Use the student as teacher and go back to step 2
        teacher_model = student_model
        

self_training_noisy_student(df_train, df_testval, df_ulbl, config.MAX_ITER)
print('DONE')