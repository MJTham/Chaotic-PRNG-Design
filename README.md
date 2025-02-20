# Image Encryption and PRNG with Logistic Map

This project implements a chaotic Pseudo-Random Number Generator (PRNG) based on the Logistic Map and uses it for image encryption. The application shuffles pixel positions and modifies grey levels using AES S-Box transformation, enhancing security and randomness.

## Features
- **Logistic Map PRNG:** Generates chaotic random sequences.
- **Image Encryption:** Uses PRNG for pixel shuffling and AES S-Box for grey-level modification.
- **Security Analysis:** Calculates Entropy, Correlation, NPCR, and UACI.
- **Randomness Validation:** Utilizes the NIST Statistical Test Suite.

---

## Prerequisites
- Python 3.x
- Required libraries listed in 'requirement.txt'
- to install the requirements, 'pip install -r requirements.txt'
- folder named <images_to_encrypt>

---

## Installation
1. **Clone this repository:**
   ```bash
   git clone https://github.com/MJTham/Chaotic-PRNG-Design.git
   cd Chaotic-PRNG-Design

## Usage
1. To generate PRNG output.
   The result (prng_output.bin) will keep into output\
   ```bash
   python lcg_prng.py
   
2.To encrypt image. Please keep your image into the folder named <images_to_encrypt>
  Once the encryption completes, all of the encrypted and decrpyted images, histogram and grayscale image will be keep in output\<image_name>_result
   ```bash
   python image_encryption.py
   
3. To run NIST test result. 
   The NIST result (NIST_test_results.txt) keep in output\ folder 
   ```bash
   python NIST_test.py
   
4. To analyse security metrics
   The analysis result (txt format) will keep within the <image_name>_result folder.
   ```bash
   python analysis.py
   
## NIST Test GitHub Main Repository
https://github.com/stevenang/randomness_testsuite

