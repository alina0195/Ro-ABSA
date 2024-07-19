import pandas as pd
import os 

path = 'df_train.csv'
df = pd.read_csv(path)
df['absa_input']= ''
df['absa_target']= ''

print(df.columns)

PROMPT_ABSA = 'Extract pairs with each category with its opinion from the following Romanian review: '

def create_df_absa_in_out(df, filename_save):
    new_df = pd.DataFrame(columns=df.columns)
    cnt = 0
    for _, row in df.iterrows():
        categories = row['all_categories_old']
        polarities = row['all_polarities']
        
        categories = [cat.strip() for cat in categories.split(';')]
        polarities = [pol.strip().lower() for pol in polarities.split(';')]
        target = []
        
        new_row = row.copy()
        new_row['absa_input'] = PROMPT_ABSA + new_row['text_cleaned'] + ' </s>'
        
        for cat, pol in zip(categories, polarities):
            current_target = cat + ' is ' + pol
            target.append(current_target)
            
        target = '; '.join(target) 
        new_row['absa_target'] = target 
        
        new_df.loc[cnt] = new_row
        cnt += 1

    new_df.drop(['all_categories','all_categories_old','all_ate','all_polarities'], inplace=True, axis=1)
    new_df.to_csv(filename_save, index=False)
    return new_df

create_df_absa_in_out(df,'df_train_absa_in_out_allLblData.csv')
