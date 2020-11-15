import pandas as pd
import random
import numpy as np

def sample(input_path,sample_num,output_path):
    df = pd.read_csv(input_path)
    index = sorted(random.sample(range(0,df.shape[0]),sample_num))
    out_df = df.loc[index]
    out_df.to_csv(output_path,index = None)

input_path = 'ratings.csv'
output_path = 'ratings_sample.csv'
sample_num = 50000
sample(input_path,sample_num,output_path)
