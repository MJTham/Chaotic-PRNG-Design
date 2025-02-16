import numpy as np
import os
import random
import time
import hashlib

def rol(n, bits, width=8):
    """Bitwise rotate left (ROL) function"""
    return ((n << bits) & (2**width - 1)) | (n >> (width - bits))

class LogisticMapPRNG:
    def __init__(self, x0=None, r=3.999):
        """
        Improved Logistic Map PRNG with dynamic seeding.
        """
        if x0 is None:
            x0 = (time.time() % 1)  # ✅ Use dynamic seed from system time
        if not (0 < x0 < 1):
            raise ValueError("Initial value x0 must be between 0 and 1")
        if not (3.57 <= r <= 4.0):
            raise ValueError("Parameter r should be between 3.57 and 4.0 for chaotic behavior")
        
        self.x = x0
        self.r = r
    
    def next(self):
        """
        Generate next value in sequence with multiple iterations to avoid periodicity.
        """
        for _ in range(10):  # ✅ Increase iterations from 5 to 10
            self.x = self.r * self.x * (1 - self.x)
        return self.x


    def chaotic_bitstream(self, n):
        """
        Generates a bitstream using the chaotic logistic map with nonlinear hashing.
        """
        bits = np.zeros(n, dtype=np.uint8)
        for i in range(n):
            self.x = self.r * self.x * (1 - self.x)
            new_x = self.r * self.x * (1 - self.x)
            
            chaotic_val = str(self.x).encode()  # Convert to bytes
            hashed_val = hashlib.sha256(chaotic_val).hexdigest()  # ✅ Apply SHA-256
            hashed_bit = int(hashed_val, 16) & 1  # ✅ Extract least significant bit
            
            bits[i] = hashed_bit
        
        return bits

    
    def generate_sequence(self, n):
        """
        Generate sequence of n pseudo-random numbers.
        """
        sequence = np.zeros(n)
        for i in range(n):
            sequence[i] = self.next()
        return sequence

def generate_prng_output(filename="prng_output.bin", num_values=100000000):  # ✅ Increase to 100M bits
    """
    Generates a sequence of pseudorandom bits using the improved Logistic Map PRNG.
    """
    output_dir = "output/prng_output"
    os.makedirs(output_dir, exist_ok=True)
    
    file_path = os.path.join(output_dir, filename)
    
    seed_x0 = (time.time() % 1)  # ✅ Dynamically set based on system time
    prng = LogisticMapPRNG(x0=seed_x0, r=3.999)
    bit_sequence = prng.chaotic_bitstream(num_values)  # ✅ Use improved chaotic bitstream
    
    with open(file_path, "wb") as f:
        np.packbits(bit_sequence).tofile(f)
    
    print(f"✅ Generated PRNG output saved to {file_path}")

# Run PRNG and generate output only if executed directly
if __name__ == "__main__":
    generate_prng_output("prng_output.bin", num_values=100000000)  # ✅ 100M bits
