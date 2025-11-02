import numpy as np
import pickle
import os
from time import time
from tensorflow.keras.models import load_model
import argparse

"""
USAGE: 

"""
# -----------------------------
# Load CIFAR-10 dataset
# -----------------------------
def load_cifar10_from_directory(directory):
    def load_batch(batch_file):
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b'data']
        labels = batch[b'labels']
        return images, labels

    # Load test batch only
    x_test, y_test = load_batch(os.path.join(directory, "test_batch"))
    
    # Reshape to (32,32,3)
    x_test = x_test.reshape(x_test.shape[0], 3, 32, 32).transpose(0, 2, 3, 1)
    
    # Convert to grayscale
    x_test_gray = np.dot(x_test[...,:3], [0.2989, 0.5870, 0.1140])
    x_test_gray = x_test_gray[..., np.newaxis]  # shape (N,32,32,1)
    
    return x_test_gray.astype('float32'), np.array(y_test)

# -----------------------------
# Parse command-line arguments
# -----------------------------
parser = argparse.ArgumentParser(description="Run inference on CIFAR-10 .h5 model")
parser.add_argument('--model', type=str, required=True, help="Path to .h5 Keras model")
parser.add_argument('--num_images', type=str, default="100",
                    help="Number of test images to run inference (or 'full' for entire test set)")
args = parser.parse_args()

# -----------------------------
# Paths
# -----------------------------
dataset_dir = '/home/root/jupyter_notebooks/PratikEmAi2025/cifar-10-batches-py/'

# -----------------------------
# Load dataset
# -----------------------------
x_test, y_test = load_cifar10_from_directory(dataset_dir)
print(f"Test samples: {x_test.shape[0]}")

# Normalize
x_test_norm = x_test / 255.0
print("Normalization complete.")

# -----------------------------
# Load model
# -----------------------------
model = load_model(args.model, compile=False)
print(f"Model '{args.model}' loaded successfully.")

# -----------------------------
# Determine number of images for inference
# -----------------------------
if args.num_images.lower() == "full":
    num_images = x_test_norm.shape[0]
else:
    num_images = min(int(args.num_images), x_test_norm.shape[0])

print(f"Running inference on {num_images} images...")

# -----------------------------
# Run inference
# -----------------------------
y_pred = []
start = time()
for i in range(num_images):
    img = np.expand_dims(x_test_norm[i], axis=0)  # shape (1,32,32,1)
    probs = model.predict(img, verbose=0)
    pred_class = np.argmax(probs)
    y_pred.append(pred_class)
stop = time()

# -----------------------------
# Metrics
# -----------------------------
y_pred = np.array(y_pred)
true_classes = y_test[:num_images]
correct = np.sum(y_pred == true_classes)
accuracy = correct / num_images * 100
total_time = stop - start
avg_time_per_image = total_time / num_images
fps = num_images / total_time

print(f"Accuracy: {accuracy:.2f}%")
print(f"Total execution time: {total_time:.4f} s")
print(f"Average inference time per image: {avg_time_per_image*1000:.2f} ms")
print(f"Throughput: {fps:.2f} FPS")
