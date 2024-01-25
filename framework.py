# Import necessary packages
import multiprocessing
from torch import nn
from pathlib import Path
from helper_functions import *
from torch.utils.data import DataLoader
from timeit import default_timer as timer
import os
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
import torch
import torchvision.io
from torchvision import datasets, transforms

""" Define several helper functions """

def walk_through_dir(dir_path):
    """
    Walks through the directory containing images
    """
    for dir_path, dir_names, filenames in os.walk(dir_path):
        print(f"There are {len(dir_names)} directories and {len(filenames)} images in '{dir_path}'.")


def show_image(images_path, seed=42):
    """
    Shows a single random image and its relevant information
    """
    # Set seed to get consistent results
    random.seed(seed)
    # Get random image path
    random_image_path = random.choice(images_path)
    # Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem
    # Open image
    img = Image.open(random_image_path)
    # Convert image into an array
    img_array = np.asarray(img)
    # Print related data
    plt.imshow(img)
    plt.axis(False)
    plt.title(f"Image class: {image_class} | Image shape: {img_array.shape}\nImage height: {img.height} | Image width: {img.width}")
    plt.show()


def plot_transformed_images(images_path, transform, seed=42):
    """
    Creates a plot that shows the effect of using a transform to convert
    images into PyTorch Tensors
    """
    random.seed(seed)
    random_image_paths = random.sample(images_path, k=3)
    with Image.open(random_image_paths[0]) as f:
        fig, ax = plt.subplots(3, 2)
        fig.set_size_inches(10, 8)
        ax[0, 0].imshow(f)
        ax[0, 0].set_title(f"Original | Size: {f.size}")
        ax[0, 0].axis("off")
        transformed_image = transform(f).permute(1, 2, 0)
        ax[0, 1].imshow(transformed_image)
        ax[0, 1].set_title(f"Transformed | Size: {transformed_image.shape}")
        ax[0, 1].axis("off")

        fig.suptitle(f"Class: {random_image_paths[0].parent.stem}", fontsize=16)
    with Image.open(random_image_paths[1]) as f2:
        ax[1, 0].imshow(f2)
        ax[1, 0].set_title(f"Original | Size: {f2.size}")
        ax[1, 0].axis("off")
        transformed_image = transform(f2).permute(1, 2, 0)
        ax[1, 1].imshow(transformed_image)
        ax[1, 1].set_title(f"Transformed | Size: {transformed_image.shape}")
        ax[1, 1].axis("off")

        fig.suptitle(f"Class: {random_image_paths[1].parent.stem}", fontsize=16)
    with Image.open(random_image_paths[2]) as f3:
        ax[2, 0].imshow(f3)
        ax[2, 0].set_title(f"Original | Size: {f3.size}")
        ax[2, 0].axis("off")
        transformed_image = transform(f3).permute(1, 2, 0)
        ax[2, 1].imshow(transformed_image)
        ax[2, 1].set_title(f"Transformed | Size: {transformed_image.shape}")
        ax[2, 1].axis("off")

        fig.suptitle(f"Class: {random_image_paths[2].parent.stem}", fontsize=16)
    plt.show()


def show_transformed_image(data, classes, seed=42):
    random.seed(seed)
    random_samples = random.sample(range(len(data)), k=1)
    targ_image, targ_label = data[random_samples[0]][0], data[random_samples[0]][1]

    targ_image_adjust = targ_image.permute(1, 2, 0)

    plt.imshow(targ_image_adjust)
    plt.axis("off")
    plt.title(f"Class: {classes[targ_label]} | Shape: {targ_image_adjust.shape}")
    plt.show()


def train_step(model, dataloader, loss_fn, optimizer, device="cpu"):
    """
    Defines the train step of the machine learning loop
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through dataloader batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def test_step(model, dataloader, loss_fn, device="cpu"):
    """
    Defines test step in the machine learning loop
    """
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item() / len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model, train_dataloader, test_dataloader, optimizer, loss_fn, epochs):
    # 2. Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": []}
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}")

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def plot_loss_curves(results):
    """
    Plots graph of loss, acc for both test and train
    """
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def transform_image(image_path, transform, device='cpu'):
    """
    Makes a prediction on an image using trained model
    """
    # Load image as a tensor and convert type into Torch float 32
    prediction_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    prediction_image = prediction_image / 255
    prediction_image_reshaped = transform(prediction_image)
    prediction_image_reshaped_batched = prediction_image_reshaped.unsqueeze(dim=0).to(device)
    return prediction_image_reshaped_batched


def make_prediction(model, classes, transformed_image):
    """
    Passes transformed images through model and prints result after evaluation
    """
    model.eval()
    with torch.inference_mode():
        model_0_predictions = model(transformed_image)
        model_0_prediction_probabilities = torch.softmax(model_0_predictions, dim=1)
        model_0_prediction_label = torch.argmax(model_0_prediction_probabilities, dim=1)
        final_prediction = classes[model_0_prediction_label]
        return final_prediction


""" The actual neural network begins below """

# Define a multiprocessing environment
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    # Create device agnostic code
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup paths to image folder
    dir_path = Path("C:\\Users\\vmath\\OneDrive\\Pictures\\Machine Learning Pictures")
    food_dir_path = dir_path / "foods"
    train_dir = food_dir_path / "train"
    test_dir = food_dir_path / "test"

    # Get all image paths
    food_dir_path_list = list(food_dir_path.glob("*/*/*.jpg"))

    # Create a transform to convert images into PyTorch tensors
    data_transform = transforms.Compose([
        # Resize the images to 64x64
        transforms.Resize(size=(64, 64)),
        # Flip the images randomly on the horizontal and augment randomly
        transforms.RandomHorizontalFlip(p=0.5),
        # Turn the image into a Tensor
        transforms.ToTensor()
    ])

    # Create a folder in PyTorch containing all images
    train_data = datasets.ImageFolder(root=str(train_dir), transform=data_transform, target_transform=None)
    test_data = datasets.ImageFolder(root=str(test_dir), transform=data_transform)

    # Turn folders into iterables that can be read by PyTorch
    train_dataloader = DataLoader(dataset=train_data, batch_size=1, num_workers=os.cpu_count(), shuffle=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=1, num_workers=os.cpu_count(), shuffle=False)
    classes = train_data.classes

    # Define your convolutional neural network
    class TinyVGG(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(in_channels=input_shape,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=hidden_units,
                          out_channels=hidden_units,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2))
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=hidden_units * 16 * 16,
                          out_features=output_shape)
            )

        def forward(self, x: torch.Tensor):
            return self.classifier(self.conv_block_2(self.conv_block_1(x)))

    # Create an instance of your model with a seed to avoid refreshing tensor values
    torch.manual_seed(42)
    model_0 = TinyVGG(input_shape=3,
                      hidden_units=32,
                      output_shape=len(train_data.classes)).to(device)

    train_dataloader_iter = iter(train_dataloader)
    img_batch, label_batch = next(train_dataloader_iter)
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]

    torch.manual_seed(42)
    epochs = 4
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(),
                                 lr=learning_rate)
    start_time = timer()
    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader,
                            test_dataloader=test_dataloader,
                            optimizer=optimizer,
                            loss_fn=loss_function,
                            epochs=epochs)
    end_time = timer()
    print(f"Total training time: {end_time-start_time:.3f} seconds")
    plot_loss_curves(model_0_results)

    # Make a prediction on a custom image
    image_details = {'path': 'C:\\Users\\vmath\\OneDrive\\Pictures\\Random Pictures\\orange_test.jpg',
                     'true_value': 'oranges'}
    image_path = image_details['path']
    image_true_value = image_details['true_value']

    # Define image transform
    prediction_image_transform = transforms.Compose([
        transforms.Resize((64, 64)),
    ])

    # Pass image through transformation function (To tensor, reshapes, to device, to correct datatype)
    prediction_image_transformed = transform_image(image_path=image_path, transform=prediction_image_transform)

    # Make an evaluation with your image
    result = make_prediction(model=model_0, transformed_image=prediction_image_transformed, classes=classes)
    if result == image_true_value:
        print(f"Image Value: {image_true_value} | Prediction Value by Model: {result} | Success: True")
    else:
        print(f"Image Value: {image_true_value} | Prediction Value by Model: {result} | Success: False")
