import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import lcg_prng as prng
import os


# Using the AES S-Box for pixel position randomization
SBOX = np.array([
    [0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76],
    [0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0],
    [0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15],
    [0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75],
    [0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84],
    [0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf],
    [0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8],
    [0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2],
    [0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73],
    [0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb],
    [0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79],
    [0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08],
    [0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a],
    [0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e],
    [0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf],
    [0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16]
])

class ImprovedImageEncryption:
    def __init__(self, x0=0.1, r=3.99):
        self.x0 = x0
        self.r = r
        self.prng = prng.LogisticMapPRNG(x0=x0, r=r)
        self.position_map = {}
        self.inverse_map = {}
        self.rgb_histograms = {}
    
    def generate_position_mapping(self, height, width):
        total_pixels = height * width
        position_map = {}
        inverse_map = {}
        rand_sequence = self.prng.generate_sequence(total_pixels * 2)
        available_positions = list(range(total_pixels))
        
        for i in range(total_pixels):
            row_idx = int(rand_sequence[i * 2] * 16) % 4
            col_idx = int(rand_sequence[i * 2 + 1] * 16) % 16
            new_pos = SBOX[row_idx][col_idx] % len(available_positions)
            new_pos = available_positions.pop(new_pos)
            position_map[i] = new_pos
            inverse_map[new_pos] = i
        
        self.position_map = position_map
        self.inverse_map = inverse_map
    
    def convert_to_grayscale(self, image):
        return np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
    
    def encrypt_image(self, image_path):
        original_img = np.array(Image.open(image_path))
        height, width = original_img.shape[:2]

        # Generate position mappings for pixel shuffling
        self.generate_position_mapping(height, width)

        # Convert image to grayscale
        grayscale_img = self.convert_to_grayscale(original_img)

        # Generate RGB histograms from original image (IMPORTANT)
        self.generate_rgb_histograms(original_img)

        # Shuffle grayscale pixels based on chaotic map
        shuffled = np.zeros_like(grayscale_img)
        for i in range(height * width):
            new_idx = self.position_map[i]
            new_i, new_j = new_idx // width, new_idx % width
            shuffled[new_i, new_j] = grayscale_img[i // width, i % width]

        return shuffled, original_img

    
    def decrypt_image(self, encrypted_image):
        height, width = encrypted_image.shape
        decrypted_gray = np.zeros_like(encrypted_image)

        # Ensure histogram mapping exists
        if not self.rgb_histograms:
            raise ValueError("RGB histograms are not initialized. Call `generate_rgb_histograms(original_image)` before decrypting.")

        # Perform decryption
        for i in range(height * width):
            if i in self.inverse_map:
                orig_idx = self.inverse_map[i]
                orig_i, orig_j = orig_idx // width, orig_idx % width
                decrypted_gray[orig_i, orig_j] = encrypted_image[i // width, i % width]

        return self.grayscale_to_rgb(decrypted_gray)

    
    def generate_rgb_histograms(self, original_image):
        height, width, _ = original_image.shape
        grayscale = self.convert_to_grayscale(original_image)

        # Ensure rgb_histograms is properly initialized
        self.rgb_histograms = {'r': {}, 'g': {}, 'b': {}}

        for i in range(height):
            for j in range(width):
                gray_value = grayscale[i, j]
                r, g, b = original_image[i, j]
                
                self.rgb_histograms['r'][gray_value] = r
                self.rgb_histograms['g'][gray_value] = g
                self.rgb_histograms['b'][gray_value] = b
    
    def get_nearest_hist_value(self, hist, gray_value):
        if gray_value in hist:
            return hist[gray_value]
        else:
            nearest_gray = min(hist.keys(), key=lambda k: abs(k - gray_value))
            return hist[nearest_gray]

    def grayscale_to_rgb(self, grayscale_image):
        height, width = grayscale_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(height):
            for j in range(width):
                gray_value = grayscale_image[i, j]
                rgb_image[i, j] = [
                    self.rgb_histograms['r'].get(gray_value, gray_value),
                    self.rgb_histograms['g'].get(gray_value, gray_value),
                    self.rgb_histograms['b'].get(gray_value, gray_value)
                ]
        
        return rgb_image
    
    def plot_histogram(self, original_img, encrypted_img, results_dir, base_name):
        """
        Saves RGB histograms for both original and encrypted images.
        Handles grayscale images by converting them to 3D (RGB format).
        """
        colors = ['red', 'green', 'blue']
        labels = ['R', 'G', 'B']

        # Convert grayscale image to 3D by duplicating channels
        if len(original_img.shape) == 2:  # Grayscale image check
            original_img = np.stack([original_img] * 3, axis=-1)

        if len(encrypted_img.shape) == 2:  # Encrypted grayscale image check
            encrypted_img = np.stack([encrypted_img] * 3, axis=-1)

        # Iterate over R, G, B channels (0, 1, 2)
        for i, color in enumerate(colors):
            plt.figure(figsize=(6, 4))
            
            # Plot original image histogram
            plt.hist(original_img[:, :, i].ravel(), bins=256, color=color, alpha=0.7, density=True)
            plt.title(f"{base_name} Original {labels[i]} Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.xlim(0, 255)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{base_name}_original_{labels[i]}_histogram.png"))
            plt.close()

            plt.figure(figsize=(6, 4))
            
            # Plot encrypted image histogram
            plt.hist(encrypted_img[:, :, i].ravel(), bins=256, color=color, alpha=0.7, density=True)
            plt.title(f"{base_name} Encrypted {labels[i]} Histogram")
            plt.xlabel("Pixel Intensity")
            plt.ylabel("Frequency")
            plt.xlim(0, 255)
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f"{base_name}_encrypted_{labels[i]}_histogram.png"))
            plt.close()

    def save_results(self, original_img, encrypted_image, decrypted_image, image_path):
        """
        Dynamically saves original, encrypted, and decrypted images into a folder inside `output/`.
        Returns the results directory path.
        """
        import os
        from PIL import Image

        # Extract the image file name without extension
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Define output directory structure
        results_dir = os.path.join("output", f"{base_name}_results")
        
        # Ensure the directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Save the images inside `output/[image_name]_results/`
        Image.fromarray(original_img.astype(np.uint8)).save(os.path.join(results_dir, f"{base_name}_original.tiff"))
        Image.fromarray(encrypted_image.astype(np.uint8)).save(os.path.join(results_dir, f"{base_name}_encrypted.png"))
        Image.fromarray(decrypted_image.astype(np.uint8)).save(os.path.join(results_dir, f"{base_name}_decrypted.tiff"))

        # Call histogram plotting method to store histograms in the same folder
        self.plot_histogram(original_img, encrypted_image, results_dir, base_name)

        print(f"Results saved in: {results_dir}")

        return results_dir  # ✅ Ensure the folder path is returned



    def process_images_in_folder(self, folder_path):
        """
        Processes all images in a given folder by performing encryption, decryption, 
        and running security analysis.
        """
        import os
        from PIL import Image
        import numpy as np
        from analysis import EncryptionAnalysis  # Import EncryptionAnalysis class

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Get all image files from the folder
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")

        print(f"Found {len(image_files)} images in '{folder_path}'. Processing...")

        analyzer = EncryptionAnalysis()  # ✅ Define the analyzer instance

        for image_name in image_files:
            image_path = os.path.join(folder_path, image_name)

            # Load image
            original_img = np.array(Image.open(image_path).convert("L"))  # Convert to grayscale

            # Encrypt image
            encrypted_img, _ = self.encrypt_image(image_path)

            # Decrypt image
            decrypted_img = self.decrypt_image(encrypted_img)

            # Save results dynamically
            results_dir = self.save_results(original_img, encrypted_img, decrypted_img, image_path)

            # ✅ Run security analysis using the defined `analyzer` object
            analyzer.analyze_image(original_img, encrypted_img, image_name, results_dir)

        print(f"All images processed successfully! Results and analysis saved in respective folders.")


if __name__ == "__main__":
    encryptor = ImprovedImageEncryption(x0=0.1, r=3.99)
    
    # Specify the folder containing images
    image_folder_path = "images_to_encrypt"

    # Run batch encryption on all images in the folder
    encryptor.process_images_in_folder(image_folder_path)
