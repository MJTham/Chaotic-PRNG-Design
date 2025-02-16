import numpy as np

def read_prng_output(filename="prng_output.bin"):
    """
    Reads and displays the first 100 bits from the PRNG output file.
    """
    try:
        with open(filename, "rb") as f:
            byte_data = np.frombuffer(f.read(), dtype=np.uint8)  # Read file as bytes
            bit_data = np.unpackbits(byte_data)  # Convert to binary (0s and 1s)
        
        print("First 100 bits of PRNG output:")
        print(bit_data[:100])  # Print only the first 100 bits
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found. Make sure it exists in the same directory.")

# Run the decoding function
if __name__ == "__main__":
    read_prng_output("prng_output.bin")
