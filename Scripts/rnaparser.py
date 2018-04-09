import numpy as np
import pandas as pd

def labelToNum(label):
      if label == 'Primary Tumor':
            return 1
      if label == 'Solid Tissue Normal':
            return 0
      if label == 'Metastatic':
            return 1
      raise ValueError('Could not find label: ' + label)

data = pd.read_csv('rnaseq_data.csv',header=0,index_col=0)
mapping = pd.read_csv('rnaseq_mapping.csv',header=0,index_col=0)
data_cols = data.columns.tolist()

features = data.T.as_matrix()
labels = np.asarray([labelToNum(mapping.loc[x][0]) for x in data_cols])