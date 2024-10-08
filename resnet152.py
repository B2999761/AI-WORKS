import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet152
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import seaborn as sns
import os
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load the dataset
dataset = torchvision.datasets.ImageFolder(root="/home/aimlgpu2/SARAN_cv/Resnet101/Resnet101_newversion/ResNet101-Implementation/normal/", transform=transform)

# Create training and validation splits
dataset_size = len(dataset)
indices = list(range(dataset_size))
train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)  # 20% for validation

# Create data samplers and loaders
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=train_sampler, num_workers=4)
val_loader = torch.utils.data.DataLoader(dataset, batch_size=16, sampler=val_sampler, num_workers=4)

# Define the model (ResNet152)
model = resnet152(pretrained=True)

# Replace the last layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(dataset.classes))

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()

# Use Adam optimizer with learning rate scheduling
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # Reduce LR after every 7 epochs

# Move the model to the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the number of epochs
num_epochs = 50  # Increased to allow more training

# Prepare to save the metrics and model outputs
output_dir = "./output_metrics"
os.makedirs(output_dir, exist_ok=True)

train_losses = []
val_losses = []
val_accuracies = []

# Train the model
for epoch in range(num_epochs):
    # Train the model on the training set
    model.train()
    train_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        # Move the data to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Update the training loss
        train_loss += loss.item() * inputs.size(0)

    # Step the learning rate scheduler
    scheduler.step()

    # Evaluate the model on the validation set
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            # Move the data to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Update the validation loss and accuracy
            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print the training and validation loss and accuracy
    train_loss /= len(train_sampler)
    val_loss /= len(val_sampler)
    val_acc = val_acc.double() / len(val_sampler)

    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

    # Save losses and accuracy for plotting later
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

# Write losses and accuracies to a text file
with open(os.path.join(output_dir, "training_stats.txt"), "w") as f:
    for epoch in range(num_epochs):
        f.write(f"Epoch {epoch + 1}:\n")
        f.write(f"Train Loss: {train_losses[epoch]:.4f}, Val Loss: {val_losses[epoch]:.4f}, Val Accuracy: {val_accuracies[epoch]:.4f}\n\n")

# Plot training and validation loss curves
plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_curves.png"))
plt.close()

# Confusion matrix (normalized for readability)
conf_matrix = confusion_matrix(all_labels, all_preds)
conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]  # Normalize

# Plot and save the confusion matrix (Only diagonal values)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_norm, annot=False, cmap="Blues", xticklabels=dataset.classes, yticklabels=dataset.classes)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Normalized)")
plt.savefig(os.path.join(output_dir, "confusion_matrix_normalized.png"))
plt.close()

print(f"Training and validation loss curves, confusion matrix, and training statistics saved in {output_dir}")
