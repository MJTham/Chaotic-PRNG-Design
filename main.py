from image_encryption import ImageEncryption
import cv2
import matplotlib.pyplot as plt

def main():
    # Load test image
    image = cv2.imread('test_image.jpg')
    
    # Initialize encryption
    encryptor = ImageEncryption(seed=12345)
    
    # Encrypt image
    encrypted = encryptor.encrypt(image)
    
    # Calculate metrics
    metrics = encryptor.calculate_metrics(image, encrypted)
    print("Security Metrics:", metrics)
    
    # Plot histograms
    encryptor.plot_histograms(image, encrypted)
    plt.show()

if __name__ == "__main__":
    main()