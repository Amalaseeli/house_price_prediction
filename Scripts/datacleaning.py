import pandas as pd
import numpy as np


def load_dataset():
   df=pd.read_csv("../data/raw/housing.csv")
   return df

#Fill missing value
def fill_missing_values(df):
    mean_value = df['total_bedrooms'].mean()
    df['total_bedrooms'] = df['total_bedrooms'].fillna(mean_value)
    return df

# label encoding
def get_dummies_data(df):
   df_encoded=pd.get_dummies(df).astype(int)
   return df_encoded
   
if __name__=="__main__":
   df=load_dataset()
   print(df.head(5))
   df=fill_missing_values(df)
#    print(df.total_bedrooms.to_list())
   df_encoded=get_dummies_data(df)
   df_encoded.set_index('longitude', inplace=True)
   print(df_encoded.columns)
   df_encoded.to_csv('../data/processed/clean_data.csv')
   
