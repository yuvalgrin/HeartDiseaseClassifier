import pandas as pd
import torch.utils.data as data
from sklearn.preprocessing import StandardScaler


# Data loader that we implemented to use the data from our CSV file
class CsvLoader(data.Dataset):
    def __init__(self, file):
        file_out = pd.read_csv(file)
        x = file_out.iloc[:, 2:31].values
        y = file_out.iloc[:, 1].values

        sc = StandardScaler()
        self.x_train = sc.fit_transform(x)
        self.y_train = pd.DataFrame(data=y)

        self.len = len(self.y_train)

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]

    def __len__(self):
        return self.len
