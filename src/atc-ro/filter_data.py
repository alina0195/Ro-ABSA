
import pandas as pd
from collections import Counter
import re
# from langdetect import detect, detect_langs
import string
from transformers import pipeline
from tqdm import tqdm 

class config:
    DF_ALL_PATH = 'augmented_reviews.csv'
    DF_ULBL_PATH = 'unlabeled_reviews.csv'
    DF_PSLBL_PATH = 'pslbl_review.csv'
    DF_ENTAILED_PATH = 'df_entailed.csv'
    

df_all = pd.read_csv(config.DF_ALL_PATH)

df_pslbl = pd.read_csv(config.DF_PSLBL_PATH)
df_ulbl = pd.read_csv(config.DF_ULBL_PATH)
df_ent = pd.read_csv(config.DF_ENTAILED_PATH)

df_from_lbl = df_all[~((df_all['data_origin'] == 'original_and_mlm') & (df_all['data_origin']=='bt_chinese'))]

print(Counter(df_all['data_origin'].values))

# 0)0. dropnans
df_pslbl.dropna(subset=['text'], inplace=True)
df_ulbl.dropna(subset=['text'], inplace=True)
df_ent.dropna(subset=['text'], inplace=True)

# 0)1. Remove emojis, links, html tags, diacritics

def clean_up_review(review):
    def remove_emojis(text):
        emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642"
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"  # dingbats
                                u"\u3030"
                                "]+", flags=re.UNICODE)

        # Replace emojis with an empty string
        text_without_emojis = emoji_pattern.sub(r'', text)

        return text_without_emojis

    try:
      # Remove urls, bullet points, extra whitespace and HTML tags
      review = remove_emojis(review)
      review = re.sub(r'https?://\S+|www\.\S+', '', review)
      review = re.sub(r'<.*?>', ' ', review)
      review = re.sub(r'(\s*[\*\•]\s*)', ' ', review)
      review = re.sub(r'\s+', ' ', review).strip()

      # Replace special characters with Romanian diacritics
      review = review.replace('&acirc;', 'â').replace('&icirc;', 'î').replace('&scedil;', 'ș').replace('&tcedil;', 'ț').replace('&Acirc;', 'Â').replace('&Icirc;', 'Î').replace('&Scedil;', 'Ș').replace('&Tcedil;', 'Ț')
      review = review.replace("ţ", "ț").replace("ş", "ș").replace("Ţ", "Ț").replace("Ş", "Ș")
      return review

    except Exception as e:
      print(f"Error {e} occured for:\n{review}")
      return None

df_from_lbl['text_cleaned'] = df_from_lbl['text'].apply(clean_up_review)
df_pslbl['text_cleaned'] = df_pslbl['text'].apply(clean_up_review)
# df_ulbl['text_cleaned'] = df_ulbl['text'].apply(clean_up_review)
df_ent['text_cleaned'] = df_ent['text'].apply(clean_up_review)

df_pslbl.dropna(subset=['text'], inplace=True)
# df_ulbl.dropna(subset=['text'], inplace=True)
df_ent.dropna(subset=['text'], inplace=True)
df_from_lbl.dropna(subset=['text'], inplace=True)


# 0)2.Filter non-romanian reviews from KB
##### only df_ulbl #####
# def is_romanian(text, threshold=0.8):
#     try:
#         lang_prob = detect_langs(text)
#         for item in lang_prob:
#             if item.lang == 'ro' and item.prob >= threshold:
#                 return True
#         return False
#     except:
#         return False

# for i in tqdm(range(100)):
#     print('Resolving Ro lang detection...')
#     df_ulbl['is_ro_lang'] = df_ulbl['text_cleaned'].apply(lambda x: is_romanian(x, 0.8))
# print('done')

# df_ulbl.to_csv('unlabeled_reviews_cleaned.csv',index=False)

# # 1) remove short reviews after cleaning
def has_more_words(review):
  try:
    length = len(review.split())
    if length < 5:
      return False
    else:
      return True
  except Exception as e:
    print(f'Exception {e} for {review}')
    return None

df_ulbl['has_more_words']=df_ulbl['text_cleaned'].apply(has_more_words)
df_pslbl['has_more_words']=df_pslbl['text_cleaned'].apply(has_more_words)
df_ent['has_more_words']=df_ent['text_cleaned'].apply(has_more_words)
df_from_lbl['has_more_words']=df_from_lbl['text_cleaned'].apply(has_more_words)

df_ulbl = df_ulbl[(df_ulbl['has_more_words']==True) & (df_ulbl['is_ro_lang']==True)]
df_pslbl = df_pslbl[df_pslbl['has_more_words']==True]
df_ent = df_ent[df_ent['has_more_words']==True]
df_from_lbl = df_from_lbl[df_from_lbl['has_more_words']==True]


# 2) remove dupicated reviews 

def remove_punct(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text


for i in tqdm(range(100)):
    print('\nResolving duplicates...')
    df_from_lbl['text_no_punct'] = df_from_lbl['text_cleaned'].apply(remove_punct)
    df_from_lbl.drop_duplicates(subset=['text_no_punct'], inplace=True)
print('done recolvng duplicates')

# 4) check if the aspect is still entailed with the review. if not, discard
for i in tqdm(range(100)):
    print('\nResolving duplicates...')
    pipeline_for_entailment = pipeline(model="facebook/bart-large-mnli")
print('done loading pipeline')

def check_entailment(pipe: pipeline, text: str, candidates: str):
    result = ''
    
    # candidates = 'calitate; marimi; produs;'
    if type(candidates)==str and len(candidates)>2:
        candidates = candidates.split('|')
        for candidate in candidates:
            if len(candidate)>2:
                pipe_res= pipe(text, candidate_labels=candidate)
                if pipe_res['scores'][0] > 0.4:
                    result+=' yes|'
                else:
                    result += ' no|'
    return result

def select_labels_based_on_entailment(labels, entailments):
    labels = labels.split(';')
    new_labels = ''
    
    if entailments!='':
        entailments = entailments.split('|')
        for lbl,entail in zip(labels, entailments):
            if entail.strip()=='yes':
                new_labels+=lbl + "; "
    return new_labels


for i in tqdm(range(100)):
    print('\nResolving entailments...')
    df_from_lbl['entailements'] = df_from_lbl.apply(lambda x: check_entailment(pipeline_for_entailment, x['text_cleaned'],x['all_categories']), axis=1)
print('done solving entailments')
    
df_from_lbl['new_labels'] = df_from_lbl.apply(lambda x: select_labels_based_on_entailment(x['all_categories','entailments']), axis=1)
df_from_lbl = df_from_lbl[df_from_lbl['new_labels']!='']

df_ent.to_csv('entainled_aspects_from_rev_cleaned.csv',index=False)
df_pslbl.to_csv('labeled_reviews_by_student_cleaned.csv',index=False)
df_from_lbl.to_csv('reviews_mlm_bt_cleaned.csv',index=False)

# 3) Add "Despre {categorie}" before each review from KB
inner_categories_mapped = {'GamingHeadset':'căști audio',
                           'Microphone':'microfon',
                           'DeskChair':'scaun de birou',
                           'Scooters':'scutere',
                           'Scooter':'scutere',
                           'Fitness_Bike':'bicicletă de sport',
                           'FitnessBike':'bicicletă de sport',
                           'TV':'televizor (TV)',
                           'Vacuum_Cleaner':'aspirator',
                           'Tripod':'trepied',
                           'ControllerAndSteeringWheel':'controler sau volan',
                           'Boots':'bocanci',
                           'Ski':'echipament de schiat'}


def add_description_in_review(row):
    if len(row['most_inner_category'])>2:
        text =  'Despre ' + inner_categories_mapped[row['most_inner_category']]+ ': ' + row['text_cleaned']
    return text

df_ulbl['text_modified'] = df_ulbl.apply(lambda x: add_description_in_review(x), axis=1)


# save results with data cleaned
df_ulbl.to_csv('unlabeled_reviews_cleaned.csv',index=False)