import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from PIL import Image
import image_encryption as encrypt
import os

class EncryptionAnalysis:
    def __init__(self):
        """Initialize the analysis tools"""
        pass
    
    def calculate_entropy(self, image):
        """
        Calculate Shannon entropy of the image
        
        Parameters:
        image (numpy.array): Input image
        
        Returns:
        float: Entropy value
        """
        _, counts = np.unique(image, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy
    
    def calculate_correlation(self, image, direction='horizontal'):
        """
        Calculate correlation coefficient between adjacent pixels
        
        Parameters:
        image (numpy.array): Input image
        direction (str): 'horizontal', 'vertical', or 'diagonal'
        
        Returns:
        float: Correlation coefficient
        """
        # Get pairs of adjacent pixels
        if direction == 'horizontal':
            x = image[:, :-1].flatten()
            y = image[:, 1:].flatten()
        elif direction == 'vertical':
            x = image[:-1, :].flatten()
            y = image[1:, :].flatten()
        else:  # diagonal
            x = image[:-1, :-1].flatten()
            y = image[1:, 1:].flatten()
        
        correlation, _ = pearsonr(x, y)
        return correlation
    
    def calculate_npcr_uaci(self, original_image, encrypted_image):
        """
        Calculate NPCR and UACI values
        
        Parameters:
        original_image (numpy.array): Original image
        encrypted_image (numpy.array): Encrypted image
        
        Returns:
        tuple: (NPCR value, UACI value)
        """
        diff_array = np.zeros_like(original_image)
        diff_array[original_image != encrypted_image] = 1
        
        # NPCR calculation
        npcr = np.sum(diff_array) * 100 / diff_array.size
        
        # UACI calculation
        uaci = np.sum(np.abs(original_image - encrypted_image)) * 100 / (255 * diff_array.size)
        
        return npcr, uaci
    
    def plot_rgb_histograms(self, image, title):
        """
        Plot RGB histograms for the given image.
        
        Parameters:
        image (numpy.array): Input image
        title (str): Title for the plot
        """
        plt.figure()
        plt.title(title)
        
        if len(image.shape) == 2:  # Grayscale image
            plt.hist(image.ravel(), 256, [0, 256], color='gray', alpha=0.7)
        else:  # Color image
            colors = ('b', 'g', 'r')
            for i, color in enumerate(colors):
                plt.hist(image[:, :, i].ravel(), 256, [0, 256], color=color, alpha=0.7)
        
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.show()
    
    def analyze_image(self, original_image, encrypted_image, image_name, results_dir):
        """
        Perform complete analysis of encryption and save results to a file.

        Parameters:
        original_image (numpy.array): Original image
        encrypted_image (numpy.array): Encrypted image
        image_name (str): Name of the image file
        results_dir (str): Directory to save results
        """
        results = {}

        # Calculate entropy
        results['original_entropy'] = self.calculate_entropy(original_image)
        results['encrypted_entropy'] = self.calculate_entropy(encrypted_image)

        # Calculate correlation coefficients
        directions = ['horizontal', 'vertical', 'diagonal']
        for direction in directions:
            results[f'original_correlation_{direction}'] = self.calculate_correlation(original_image, direction)
            results[f'encrypted_correlation_{direction}'] = self.calculate_correlation(encrypted_image, direction)

        # Calculate NPCR and UACI
        npcr, uaci = self.calculate_npcr_uaci(original_image, encrypted_image)
        results['npcr'] = npcr
        results['uaci'] = uaci

        # Save results to a file
        result_file_path = os.path.join(results_dir, f"{image_name}_encryption_analysis.txt")
        with open(result_file_path, "w") as f:
            f.write("Encryption Security Analysis Report\n")
            f.write("=================================\n\n")
            f.write(f"Original Image Entropy: {results['original_entropy']:.4f}\n")
            f.write(f"Encrypted Image Entropy: {results['encrypted_entropy']:.4f}\n\n")
            f.write("Correlation Coefficients:\n")
            for direction in directions:
                f.write(f" - Original {direction}: {results[f'original_correlation_{direction}']:.4f}\n")
                f.write(f" - Encrypted {direction}: {results[f'encrypted_correlation_{direction}']:.4f}\n")
            f.write(f"\nNPCR: {results['npcr']:.4f}%\n")
            f.write(f"UACI: {results['uaci']:.4f}%\n")

        print(f"Analysis results saved to: {result_file_path}")
        return results

# Example usage
if __name__ == "__main__":
    # Initialize analysis tools
    analyzer = EncryptionAnalysis()
    
    # Load original and encrypted images
    image_path = "Figure_1.png"  # Replace with your image path
    original_img = np.array(Image.open(image_path).convert('L'))
    encryptor = encrypt.ImageEncryption(x0=0.1, r=3.99)
    encrypted_img = encryptor.encrypt_image(image_path)
    
    # Perform analysis
    results = analyzer.analyze_image(original_img, encrypted_img)
    
    # Print results
    print("\nAnalysis Results:")
    print("================")
    print(f"Original Image Entropy: {results['original_entropy']:.4f}")
    print(f"Encrypted Image Entropy: {results['encrypted_entropy']:.4f}")
    print("\nCorrelation Coefficients:")
    for direction in ['horizontal', 'vertical', 'diagonal']:
        print(f"Original Image {direction}: {results[f'original_correlation_{direction}']:.4f}")
        print(f"Encrypted Image {direction}: {results[f'encrypted_correlation_{direction}']:.4f}")
    print(f"\nNPCR: {results['npcr']:.4f}%")
    print(f"UACI: {results['uaci']:.4f}%")
    
    # Plot histograms
    analyzer.plot_rgb_histograms(original_img, "Original Image")
    analyzer.plot_rgb_histograms(encrypted_img, "Encrypted Image")