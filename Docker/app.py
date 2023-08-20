import sys
import numpy as np

def main():
    if len(sys.argv) != 2:
        print("Usage: python app.py <number>")
        return

    try:
        number = float(sys.argv[1])
    except ValueError:
        print("Invalid input. Please provide a valid number.")
        return

    square_root = np.sqrt(number)
    print(f"The square root of {number} is {square_root:.2f}")

if __name__ == "__main__":
    main()
