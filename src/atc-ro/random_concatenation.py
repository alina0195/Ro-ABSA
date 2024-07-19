import pandas as pd
from collections import Counter 


path = 'data/df_train.csv'
df = pd.read_csv(path)

all_categ = []
for idx, row in df.iterrows():
  cat = row['all_categories'].split(';')
  cat = [c.strip() for c in cat]
  all_categ.extend(cat)

freq_categories = Counter(all_categ)
freq_labels = Counter(df['all_categories'])

rare_categories = [item[0] for item in freq_categories.most_common(len(freq_categories))]   # ['product', 'shop diversity', 'staff competency', 'shop organization', 'service', 'quality', 'price', 'environment', 'staff availability', 'delivery', 'misc', 'tech support', 'promotions', 'return warranty', 'security', 'accessibility']
# rare_categories = rare_categories[-6:]  # ['misc', 'tech support', 'promotions', 'return warranty', 'security', 'accessibility']
rare_categories = ['shop diversity', 'staff competency', 'shop organization', 'service', 'environment', 'staff availability', 'delivery', 'misc', 'tech support', 'promotions', 'return warranty', 'security', 'accessibility']


def to_keep_rare(row):
    labels = row['all_categories'].split(';')
    labels = [lbl.strip() for lbl in labels]
    
    for rare_cat in rare_categories:
        if rare_cat in labels:
            return 'add'
    return 'not_add'

max_len = 100

def is_short(row):
    text_len = len(row['text_cleaned'].split(' '))
    if text_len < max_len:
        return 'short'
    else:
        return 'not_short'        


def get_concat_instances(df):
    new_rows = []
    for i in range(len(df)):
        first_row = df.iloc[i]
        second_row = df.iloc[len(df)-i-1]
        new_row = {}
        new_row['id'] = first_row['id'] + second_row['id']
        new_row['text_cleaned'] = first_row['text_cleaned']+'. ' + second_row['text_cleaned']
        new_row['text'] = first_row['text']+'. ' + second_row['text']
        
        new_row['all_ate'] = first_row['all_ate']+'; ' + second_row['all_ate']
        new_row['all_categories'] = first_row['all_categories']+'; ' + second_row['all_categories']
        new_row['all_categories_old'] = first_row['all_categories_old']+'; ' + second_row['all_categories_old']
        
        new_row['all_polarities'] = first_row['all_polarities']+'; ' + second_row['all_polarities']
        new_row['data_origin'] = 'concat'
        new_rows.append(new_row)
    df_to_add = pd.DataFrame(new_rows)
    df_to_add['keep'] = 'add'
    df_to_add['short'] = 'not_short'
    return df_to_add


df['keep'] = df.apply(lambda x: to_keep_rare(x), axis=1)
df['short'] = df.apply(lambda x: is_short(x), axis=1)

df_selected_for_concat = df[(df['keep']=='add') & (df['short']=='short')]
df_with_concat_rev = get_concat_instances(df_selected_for_concat)
df_with_concat_rev.drop_duplicates(subset=['text_cleaned'], inplace=True)

df_rare_lbls = df[(df['keep']=='add')]

df.drop(columns=['keep','short'], inplace=True)
df_rare_lbls.drop(columns=['keep','short'], inplace=True)
df_with_concat_rev.drop(columns=['keep','short'], inplace=True)


print(f'df rare lbls columns: {df_rare_lbls.columns}')
print(f'df concat columns: {df_with_concat_rev.columns}')
print(f'df columns: {df.columns}')

df = pd.concat([df, df_with_concat_rev], axis=0, ignore_index=True)
df_rare_lbls.to_csv('df_rare_lbls.csv',index=False)
df.to_csv('df_train-augmented.csv',index=False)