import os, csv, gzip
import numpy as np

MODE = 'train'

# Get paths for the input files
path_labels = os.path.join(os.getcwd(), f'{MODE}-labels-idx1-ubyte.gz')
path_images = os.path.join(os.getcwd(), f'{MODE}-images-idx3-ubyte.gz')

print(f'Loading files from\n  {path_labels=}\n  {path_images=}')

# Load the data from the input files
with gzip.open(path_labels, 'rb') as f:
    labels: np.ndarray = np.frombuffer(f.read(), dtype=np.uint8,
                           offset=8)
with gzip.open(path_images, 'rb') as f:
    images: np.ndarray = np.frombuffer(f.read(), dtype=np.uint8,
                           offset=16).reshape(len(labels), 784)

print(f'Loaded {images.shape=}, {labels.shape=}...')
print("Writing to file.")

# Transform the input data
output_arr = []
for i in range(len(labels)):
    output_arr.append(np.insert(images[i], 0, labels[i]).tolist())

with open('fashion-mnist.csv', 'w') as csvf:
    writer = csv.writer(csvf)
    writer.writerows(output_arr)
print("Done.")

