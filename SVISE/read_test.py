import pickle
import pandas as pd
try:
    with open('SVISE/synthetic_dataset_validation/synthetic_with_wiener.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Standard pickle worked! Type:", type(data))
    if isinstance(data, dict):
        print(data.keys())
except Exception as e:
    print("Standard pickle failed:", e)

try:
    data = pd.read_pickle('SVISE/synthetic_dataset_validation/synthetic_with_wiener.pkl')
    print("Pandas read_pickle worked! Type:", type(data))
except Exception as e:
    print("Pandas read_pickle failed:", e)
