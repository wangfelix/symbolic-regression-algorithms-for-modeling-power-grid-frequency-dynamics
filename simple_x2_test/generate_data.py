import numpy as np
import os

def generate():
    print("Generating data for y = x^2...")
    # Generate 100 points between -10 and 10
    x = np.linspace(-10, 10, 100)
    y = x**2
    
    # AI Feynman expects columns x1, x2, ..., y
    data = np.column_stack((x, y))
    
    # Save to data.txt
    np.savetxt("data.txt", data)
    print("Data saved to data.txt")

if __name__ == "__main__":
    generate()
