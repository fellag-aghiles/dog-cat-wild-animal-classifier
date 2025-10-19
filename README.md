
# Description
the architectuer of the model is, it use activation function and optimisation of, and it consiste of

This project is a image classification pipeline built with PyTorch, 
It allows you to train a Convolutional Neural Network (CNN) from scratch, evaluate its performance, and use a simple GUI to classify new images interactively.
it was trainned on an overal of 16k images of cats dogs and wild animals

# What This Project Does

Automatically loads dataset from folders (each folder = one class)
Encodes labels with scikit-learn
Splits data into training (70%), validation(15%), and test(15%) sets
Defines a custom CNN model with PyTorch
Trains and evaluates the model
Plots accuracy and loss over time
Saves and reloads trained models
Includes a Tkinter-based GUI for interactive image prediction
Supports GPU acceleration (CUDA) if available

# Model Architecture

Convolutional Layers:

Conv1: 3 input channels → 32 filters (3x3 kernel, padding=1)

Conv2: 32 → 64 filters (3x3 kernel, padding=1)

Conv3: 64 → 128 filters (3x3 kernel, padding=1)

Each layer is followed by ReLU activation and 2x2 MaxPooling.


Fully Connected Layers:

Flatten layer to convert features into a vector

Linear(128 * 16 * 16 → 128)

Linear(128 → 10)


Activation Functions:

ReLU used for hidden layers

Softmax applied implicitly through CrossEntropyLoss during training



# Optimizer and Hyperparameters

Optimizer: Adam

Learning Rate: 0.0007

Loss Function: CrossEntropyLoss

Batch Size: 256

Epochs: 7


# Trainning

Training process includes:

Forward pass through the model

Loss computation

Backpropagation

Parameter update with Adam

Validation after each epoch

Results printed at each step

After training, loss and accuracy graphs are displayed for analysis.



# Dataset Setup

Your dataset should be structured as follows:
train/

  cat/
       image1.jpg
       image2.jpg
       ...

  dog/
       image1.jpg
       image2.jpg
       ...
  wild/
       image1.jpg
       image2.jpg
       ...

# How to Run

Place your dataset folder named "train" in the same directory as the script.

Open a terminal and run:
python3 Image.py

You’ll be asked:
Do you want to Train a new model or no? y/n
Type “y” to train a new model
Type “n” to load an existing model (saved_model_complete.pth)


if you choose y to train this what the result will be : 
<img src="Readme_images\trainning_logs.png">

Afterward, a Tkinter window will open for image prediction.
<img src="Readme_images\tochoosefile.png">

then you will choose a image (jpg, png, jepg)
<img src="Readme_images\choosetestimage.png">

The program displays the image along with the predicted class name
<img src="Readme_images\predictoin_result.png">

after you finish your test and close your the tkinter window, a graph that compare between trainning and validation (loss/accuracy) will show to see wehere the overfiting is and change Hyperparameters according to what we see
<img src="Readme_images\graph.png">


# Customization 

You can easily customize key parts of the code:

Change input image size:
transforms.Resize((128, 128))

Modify CNN architecture:
Edit the ImageModel class

Adjust hyperparameters:
LR = 0.0007
BATCH_SIZE = 256
EPOCHS = 7

Change dataset path:
dir_path = "train"

# Notes

If you skip training, make sure “saved_model_complete.pth” exists.
Very large batch sizes may cause GPU memory errors.
The Tkinter GUI might not run on server or google-collab

# Requirenments

PyTorch
Torchvision
scikit-learn
pandas
matplotlib
Pillow 
tkinter 
torchsummary