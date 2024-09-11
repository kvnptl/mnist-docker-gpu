# A Simple MNIST Classifier + Docker + NVIDIA GPU

<p align="center">
  <img src="https://github.com/user-attachments/assets/e5e1dd42-5ab0-484b-8bea-001bef1b9a1e" />
</p>

This repository contains a simple **MNIST classifier** built using **PyTorch**, wrapped in a Docker container with **NVIDIA GPU** support for fast training and testing. The project includes:

- **Training** and **testing scripts** for the MNIST dataset.
- **Dockerfile** with instructions for building and running the Docker image with GPU acceleration.
- Code to **visualize predictions** and save them as images.
- Setup for leveraging **NVIDIA GPUs** via Docker's GPU support, making model training significantly faster.

## Requirements

1. **Docker**: Ensure Docker is installed on your system.
   - [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
   
2. **NVIDIA Drivers**: Verify that your NVIDIA GPU is accessible on the system.
   - Run the `nvidia-smi` command to check if your GPU is recognized. Make sure to note down the **CUDA version**.
   
3. **NVIDIA Container Toolkit**: This toolkit allows Docker to use the GPU on your host machine.
   - [Install NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
   > Docker alone cannot access the GPU. The NVIDIA Container Toolkit provides the necessary drivers and runtime configurations that enable containers to leverage GPU compute power.

4. **Update CUDA Version in Dockerfile**: Update the CUDA version in the [`Dockerfile`](https://github.com/kvnptl/mnist-docker-gpu/blob/eaf381c0de8b2d3be1eb03ea662a1ff3132d9d24/Dockerfile#L1) to match (or be lower than) the CUDA version from your system.
   - Find available versions on the [NVIDIA CUDA Docker Hub](https://hub.docker.com/r/nvidia/cuda/tags?page_size=&ordering=&name=).
   - For example, if your system CUDA version is `12.2`, update the [`Dockerfile`](https://github.com/kvnptl/mnist-docker-gpu/blob/eaf381c0de8b2d3be1eb03ea662a1ff3132d9d24/Dockerfile#L1) to:
     ```dockerfile
     FROM nvidia/cuda:12.2.2-runtime-ubuntu22.04
     ```

## How to Use

1. **Build the Docker image** using the provided Dockerfile:

    ```bash
    docker build -t mnist-classifier .
    ```

2. **Run the container** in interactive mode with a GPU-enabled machine:

   (with gpu, interactive mode, local mounted volume, name)
    ```bash
    docker run --gpus all -it -v ./output:/app/output --name mnist-classifier mnist-classifier bash
    ```

    or without interactive mode (with gpu, local mounted volume, remove container after exit, in detached mode, name):
    ```bash
    docker run --gpus all -v ./output:/app/output --rm -d --name mnist-classifier mnist-classifier
    ```

4. **Visualize** the predictions in the `output` directory and also stores the model checkpoint.

## License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
