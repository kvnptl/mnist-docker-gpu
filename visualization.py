import torch
import matplotlib.pyplot as plt
import numpy as np


# Function to visualize predictions
def visualize_predictions(model, dataloader, device):
    model.eval()
    images, labels = next(iter(dataloader))
    images, labels = images.to(device), labels.to(device)

    # Get model predictions
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)

    # Plot a 3x3 grid of images with predictions and ground truth
    fig, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.flatten()
    
    for i in range(9):
        img = images[i].cpu().numpy().squeeze()  # Move image to CPU and remove unnecessary dimensions
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"GT: {labels[i].item()} | Pred: {predicted[i].item()}")
        axes[i].axis("off")  # Turn off axis labels/ticks

    plt.tight_layout()
    plt.savefig("/app/output/predictions.png")
    print(f"[INFO] Saved predictions to predictions.png")
