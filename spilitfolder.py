import os
import shutil
import random

def organize_images_and_labels(image_dir, label_dir, output_base_dir, train_ratio=0.8):
    """
    Organize images and labels into separate training and validation directories.
    Images without labels are moved to a background directory.
    
    Parameters:
        image_dir (str): Path to the directory containing images.
        label_dir (str): Path to the directory containing labels.
        output_base_dir (str): Base directory to create train/val structure.
        train_ratio (float): Ratio of images to be used for training (default is 0.8).
    """
    # Paths for training and validation directories
    train_image_dir = os.path.join(output_base_dir, 'train', 'images')
    train_label_dir = os.path.join(output_base_dir, 'train', 'labels')
    val_image_dir = os.path.join(output_base_dir, 'val', 'images')
    val_label_dir = os.path.join(output_base_dir, 'val', 'labels')
    background_dir = os.path.join(output_base_dir, 'background_images')

    # Create directories if they don't exist
    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    os.makedirs(background_dir, exist_ok=True)

    # List all images
    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    for image_name in images:
        image_path = os.path.join(image_dir, image_name)
        label_name = os.path.splitext(image_name)[0] + '.txt'
        label_path = os.path.join(label_dir, label_name)

        if os.path.exists(label_path):
            if random.random() < train_ratio:
                # Copy image and label to the training directory
                shutil.copy(image_path, train_image_dir)
                shutil.copy(label_path, train_label_dir)
            else:
                # Copy image and label to the validation directory
                shutil.copy(image_path, val_image_dir)
                shutil.copy(label_path, val_label_dir)
        else:
            # Move images without labels to the background directory
            shutil.copy(image_path, background_dir)

# Example usage
organize_images_and_labels(
    image_dir='./../images',
    label_dir='./../labels',
    output_base_dir='output'
)
