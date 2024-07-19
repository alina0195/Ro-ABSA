import pandas as pd
import os 

path = 'df_train.csv'
df = pd.read_csv(path)
df['category']= ''
df['polarity']= ''
df['alsa_input']=''
print(df.columns)

PROMPT_ALSA = 'Polarity opinion about the aspect term <{aspect}> from the following Romanian review: '

def create_df_with_one_pair(df, filename_save):
    new_df = pd.DataFrame(columns=df.columns)
    cnt = 0
    for _, row in df.iterrows():
        categories = row['all_categories_old']
        polarities = row['all_polarities']
        
        categories = [cat.strip() for cat in categories.split(';')]
        polarities = [pol.strip().lower() for pol in polarities.split(';')]
        
        for cat, pol in zip(categories, polarities):
            new_row = row.copy()
            new_row['polarity'] = pol
            new_row['category'] = cat
            new_row['alsa_input'] = PROMPT_ALSA.format(aspect=cat) + new_row['text_cleaned']
            new_df.loc[cnt] = new_row
            cnt += 1

    new_df.drop(['all_categories','all_categories_old','all_ate','all_polarities'], inplace=True, axis=1)
    new_df.to_csv(filename_save, index=False)
    return new_df

# create_df_with_one_pair(df_test,'df_testval_one_pair.csv')
create_df_with_one_pair(df,'df_train_one_pair_allLblData.csv')
