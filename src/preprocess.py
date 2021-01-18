import pandas as pd
import numpy as np

def data_process_X(df_X):
    ## data process for X data
    df_X = df_X.copy()
    _columns = [col for col in df_X.columns if "call" not in col]
    
    ## transpose
    df_res = pd.DataFrame(df_X, columns=_columns)

    df_res = df_res.T
    df_res.columns = df_res.iloc[1, :]

    df_res = df_res.drop(["Gene Description", "Gene Accession Number"], axis = 0).apply(pd.to_numeric)
    return df_res