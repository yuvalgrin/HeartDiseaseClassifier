import FullyConnected
import RandomForest
import SVM
import XGBoost
from CsvLoader import CsvLoader

if __name__ == '__main__':
    file = 'simulated HF mort data for GMPH (1K) final.csv'
    data_set = CsvLoader(file)

    # RandomForest.main(data_set)
    XGBoost.main(data_set)
    # SVM.main(data_set)
    # FullyConnected.main(data_set)
