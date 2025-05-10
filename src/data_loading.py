import pandas as pd

def load_file(path):
    data_frame = pd.read_csv(path)
    return data_frame