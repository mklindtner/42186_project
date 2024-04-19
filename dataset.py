import os
import pandas as pd
from clean_data import clean_data

github_file_url = 'https://github.com/gsverhoeven/fumbbl_datasets/blob/main/datasets/v0.6/v0.6_fumbbl_data_csv.zip.zip'

class fumbbl:
    
    def __init__(self, path='data', train_test_split=None):
        self.path = path
        self.set_data()

        if train_test_split:
            self.train_test_split = train_test_split
            self.train = self.df.sample(frac=train_test_split)
            self.test = self.df.drop(self.train.index)


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        return self.df.iloc[idx]


    def set_data(self):
        try:
            self.df = pd.read_csv(os.path.join(self.path, 'df_matches_clean.csv'), sep=',')
        except:
            print('Failed to read data file\nProceeding to download data from' + github_file_url + '\nThis may take a while...')
            self.download_data()
            clean_data()
            self.df = pd.read_csv(self.path, sep=',')
    

    def download_data(self):
        assert False, 'Not implemented yet'

