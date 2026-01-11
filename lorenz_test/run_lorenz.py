import aifeynman
import os

def run_test():
    print("Running AI Feynman on Lorenz data...")
    
    filename = "lorenz_z_data.txt"
    if not os.path.exists(filename):
        print(f"Error: {filename} not found.")
        return

    # Use 14ops.txt from library defaults (assuming user is correct and it works by magic or passing just string)
    # The user instruction was: "Just use it as function parameter. The text file is included in the aifeynman library"
    # Usually this means we pass '14ops.txt' and maybe aifeynman handles looking it up.
    
    try:
        # Increase polyfit_deg maybe? Lorenz has xy which is deg 2.
        # Arguments: pathdir, filename, BF_try_time, BF_ops_file_type
        aifeynman.run_aifeynman('./', filename, 60, "14ops.txt", polyfit_deg=3, NN_epochs=500)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_test()
