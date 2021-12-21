from chemml.datasets import load_organic_density

import warnings
warnings.filterwarnings('ignore')

molecules, target, dragon_subset = load_organic_density()
molecules.to_csv('data/molecules.csv', index=False)
target.to_csv('data/target.csv', index=False)
dragon_subset.to_csv('data/dragon_subset.csv', index=False)

print('============= Successfully Saved the Data ============')
