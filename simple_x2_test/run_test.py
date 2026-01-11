import aifeynman
import os
import sys

def run_test():
    print("Running AI Feynman test on data.txt...")
    
    # Check if data.txt exists
    if not os.path.exists("data.txt"):
        print("Error: data.txt not found. Run generate_data.py first.")
        return
    
    try:
        # Calling run_aifeynman
        # Arguments: pathdir, filename, BF_try_time, BF_ops_file_type
        # Using 60 seconds try time
        aifeynman.run_aifeynman('./', 'data.txt', 60, "14ops.txt", polyfit_deg=3, NN_epochs=500)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
