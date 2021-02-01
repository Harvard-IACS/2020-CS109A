

def RandomUniverse(df):
    df_bootstrap = df.sample(len(df), replace = True)
    return df_bootstrap