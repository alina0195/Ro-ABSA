import pandas as pd
import torch
import numpy as np
import random
import os

from torch.utils.data import DataLoader
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM)
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
    
    DATASET_RO_PATH =  ROOT + os.sep +'data/df_rare_lbls.csv'
    DATASET_SAVE_PATH =  ROOT + os.sep +'data/df_almostRare_lbls.csv'
    BATCH_SIZE = 16
    DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

set_seed(config.SEED)
print("Loading tokenizers")
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
print("Loading model")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M")
model=model.to(config.DEVICE)


def translate_batch(text_list, tokenizer_src, src_lang, tgt_lang):
    tokenizer_src.src_lang = src_lang
    inputs = tokenizer_src(text_list, padding=True, truncation=True, max_length=512, return_tensors="pt").to('cuda')
    with torch.no_grad():
        generated_tokens = model.generate(**inputs, max_length=512,
                                          forced_bos_token_id=tokenizer_src.lang_code_to_id[tgt_lang],
                                          num_beams=5, do_sample=True, early_stopping=True)
    decoded_tokens = [tokenizer_src.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_tokens]
    return [d for d in decoded_tokens]



def backtranslation_batch(text_list, tokenizer_src, tokenizer_tgt, tokenizer_tgt2, src_lang, tgt_lang, tgt_lang2=None):
    translated_text = translate_batch(text_list, tokenizer_src, src_lang, tgt_lang)
    if tgt_lang2==None:
        text_bt = translate_batch(translated_text,tokenizer_tgt, tgt_lang, src_lang)
    else:
        translated_text = translate_batch(translated_text, tokenizer_tgt, tgt_lang, tgt_lang2)
        text_bt = translate_batch(translated_text, tokenizer_tgt2, tgt_lang2, src_lang)
    return text_bt



class BTDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        item = self.df.iloc[idx]
        text = item['text_cleaned']
        return {
            'text': text,
        }


def get_dataloader(df):
    ds = BTDataset(df)
    dl = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4)
    return dl


def run_backtranslation(model, dataloader, tokenizer_src, tokenizer_tgt, tokenizer_tgt2, src_lang='ron_Latn', tgt_lang='eng_Latn', tgt_lang2 = 'zho_Hans'):
    print("Running Backtranslation...")

    model.eval()
    new_instances = []
    for step, batch in enumerate(dataloader):

        text_list = batch['text']
        if tgt_lang2!=None:
            text_list_bt = backtranslation_batch(text_list=text_list, 
                                                 tokenizer_src=tokenizer_src,
                                                 tokenizer_tgt=tokenizer_tgt,
                                                 tokenizer_tgt2= tokenizer_tgt2,
                                                 src_lang=src_lang,
                                                 tgt_lang=tgt_lang,
                                                 tgt_lang2=tgt_lang2
                                                 )
        else:
            text_list_bt = backtranslation_batch(text_list=text_list, 
                                        tokenizer_src=tokenizer_src,
                                        tokenizer_tgt=tokenizer_tgt,
                                        tokenizer_tgt2 = None,
                                        src_lang=src_lang,
                                        tgt_lang=tgt_lang,
                                        tgt_lang2=None
                                        )
        new_instances.extend(text_list_bt)
        if step % 40 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(dataloader)))
            print(text_list_bt)
            print('')
    return new_instances


df = pd.read_csv(config.DATASET_RO_PATH)
df = df[['id','text_cleaned','all_ate','all_categories','all_polarities']]
dataloader = get_dataloader(df)

bt_text_2chain = run_backtranslation(model=model,
                                     dataloader=dataloader,
                                     tokenizer_src=tokenizer,
                                     tokenizer_tgt=tokenizer,
                                     tokenizer_tgt2=tokenizer,
                                     )
bt_text_1chain = run_backtranslation(model=model,
                                     dataloader=dataloader,
                                     tokenizer_src=tokenizer,
                                     tokenizer_tgt=tokenizer,
                                     tokenizer_tgt2=None,
                                     src_lang='ron_Latn',
                                     tgt_lang='eng_Latn',
                                     tgt_lang2=None
                                     )

df['text_bt_RoEnChRo'] = bt_text_2chain
df['text_bt_RoEnRo'] = bt_text_1chain

df.to_csv(config.DATASET_SAVE_PATH, index=False)
