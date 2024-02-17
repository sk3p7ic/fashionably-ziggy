import os
import sys
import gzip
import numpy as np


def prepare_dataset(dset_mode: str) -> None:
    cwd = os.getcwd()
    # Get paths for the input files
    path_labels = os.path.join(cwd, f"{dset_mode}-labels-idx1-ubyte.gz")
    path_images = os.path.join(cwd, f"{dset_mode}-images-idx3-ubyte.gz")

    print(f"Loading files from\n\t{path_labels=}\n\t{path_images=}")

    # Load the data from the input files
    with gzip.open(path_labels, "rb") as f:
        labels: np.ndarray = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    with gzip.open(path_images, "rb") as f:
        images: np.ndarray = np.frombuffer(f.read(), dtype=np.uint8, offset=16).reshape(
            len(labels), 784
        )

    print(f"Loaded {images.shape=}, {labels.shape=}...")
    output_filename = f"fashion-mnist-{dset_mode}.dataset"
    print(f"Wrtiting to '{output_filename=}'...")

    dataset = np.insert(images, 0, labels, 1).astype("uint8")
    dataset.tofile(output_filename)
    print(f"Wrote {dataset.nbytes} bytes to {output_filename=}")


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print(f"Usage: {sys.argv[0]} [train|test]")
        sys.exit(1)
    mode = sys.argv[1].lower()
    prepare_dataset(mode)
