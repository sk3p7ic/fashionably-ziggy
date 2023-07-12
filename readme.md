# Fashionably Ziggy

_A neural network written in Zig to classify fashion items from the Fashion MNIST dataset._

<details><summary>Table of Contents</summary><p>

- [About this Project](#about-this-project)
- [License](#license)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Data Preparation](#data-preparation)

</p></details>

## About this Project

This project is a neural network written in Zig to classify fashion items from the Fashion MNIST dataset. It is a work in progress. The [Fashion MNIST dataset](https://github.com/zolandoresearch/fashion-mnist) is a drop-in replacement for the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) of handwritten digits. It is intended to serve as a direct comparison for machine learning algorithms.

This project is a work in progress. The goals of this project are:

- To implement a neural network from scratch in Zig.
- To learn more about the internals of neural networks.
- To serve as a supplement to the Machine Learning and Artificial Intelligence course that I am taking at Texas A&M University - San Antonio.
- To improve my Zig skills.

## Usage

### Prerequisites

- [Zig](https://ziglang.org/) >= 0.11.0
- [train-images-idx3-ubyte.gz](https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-images-idx3-ubyte.gz)
- [train-labels-idx1-ubyte.gz](https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/train-labels-idx1-ubyte.gz)

### Data Preparation

The Fashion MNIST dataset is not included in this repository. You must download the dataset and extract the files to the `dataset` directory. The `dataset` directory should contain the following files:

- `train-images-idx3-ubyte.gz`
- `train-labels-idx1-ubyte.gz`

The links to the files are provided in the [Prerequisites](#prerequisites) section.

Next, run the following command to install the prerequisites for the data preparation script:

```bash
pip3 install -r requirements.txt
```

Finally, run the following command to prepare the data:

```bash
python3 download.py
```

This script currently extracts the compressed files into `numpy.ndarray` objects, prepends the labels to each image (row), and then saves the list of images to `fashion-mnist.csv` in the `dataset` directory.
This is the file that the neural network will read from. Do not delete or modify this file.

## License

The MIT License (MIT) Copyright &copy; 2023 Joshua Ibrom (sk3p7ic), https://joshuaibrom.com

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
