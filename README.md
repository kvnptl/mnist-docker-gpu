# A Simple MNIST Classifier + Docker + NVIDIA GPU

This repository contains a simple **MNIST classifier** built using **PyTorch**, wrapped in a Docker container with **NVIDIA GPU** support for fast training and testing. The project includes:

- **Training** and **testing scripts** for the MNIST dataset.
- **Dockerfile** with instructions for building and running the Docker image with GPU acceleration.
- Code to **visualize predictions** and save them as images.
- Setup for leveraging **NVIDIA GPUs** via Docker's GPU support, making model training significantly faster.

## Features

- **Train** and **test** an MNIST classifier on GPUs with ease.
- **Visualize predictions**: The model outputs predictions and saves them as an image for easy inspection.
- **Dockerized** for portability and reproducibility: Easily run the project in any environment with NVIDIA GPU support.

## How to Use

1. **Build the Docker image** using the provided Dockerfile:

    ```bash
    docker build -t mnist-classifier .
    ```

2. **Run the container** and train the model on your GPU-enabled machine:
    (with gpu, interactive mode, local mounted volume, name)
    ```bash
    docker run --gpus all -it -v ./output:/app/output --name mnist-classifier mnist-classifier bash
    ```

    or without interactive mode (with gpu, local mounted volume, remove container after exit, in detached mode, name):
    ```bash
    docker run --gpus all -v ./output:/app/output --rm -d --name mnist-classifier mnist-classifier
    ```

3. **Visualize** the predictions in the `output` directory and also stores the model checkpoint.

## Requirements

- **NVIDIA GPU** with Docker and NVIDIA Container Toolkit installed.
- **PyTorch** and other dependencies are automatically installed in the container.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
