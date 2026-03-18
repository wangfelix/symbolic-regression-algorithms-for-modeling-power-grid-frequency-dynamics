import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from svise.run_analysis_5min_hyperparameter_tuning import load_data, get_valid_chunks_9_to_10
data = load_data('dataset/Frequency_data_SK.pkl')
chunks = get_valid_chunks_9_to_10(data)
print("TOTAL CHUNKS FOR TUNING:", len(chunks))
