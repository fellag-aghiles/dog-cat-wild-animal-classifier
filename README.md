
this model is trainned on 5650 cat, 5240 dog, and 5240 wild animals ( lion, tiger, fox,...)


the architectuer of the model is, it use activation function and optimisation of, and it consiste of

trainning output example :

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 32, 128, 128]             896
         MaxPool2d-2           [-1, 32, 64, 64]               0
              ReLU-3           [-1, 32, 64, 64]               0
            Conv2d-4           [-1, 64, 64, 64]          18,496
         MaxPool2d-5           [-1, 64, 32, 32]               0
              ReLU-6           [-1, 64, 32, 32]               0
            Conv2d-7          [-1, 128, 32, 32]          73,856
         MaxPool2d-8          [-1, 128, 16, 16]               0
              ReLU-9          [-1, 128, 16, 16]               0
          Flatten-10                [-1, 32768]               0
           Linear-11                  [-1, 128]       4,194,432
           Linear-12                    [-1, 3]             387
================================================================
Total params: 4,288,067
Trainable params: 4,288,067
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 10.75
Params size (MB): 16.36
Estimated Total Size (MB): 27.30

Estimated Total Size (MB): 27.30
----------------------------------------------------------------
Epoch: 1/7 | Train Loss: 0.0435 | Train Accuracy: 53.5736
          | Val Loss: 0.0066 | Val Accuracy: 71.5289

==============================
Epoch: 2/7 | Train Loss: 0.0224 | Train Accuracy: 79.7804
          | Val Loss: 0.0046 | Val Accuracy: 82.1074

==============================
Epoch: 3/7 | Train Loss: 0.0143 | Train Accuracy: 88.079
          | Val Loss: 0.0043 | Val Accuracy: 84.6694

==============================
Epoch: 4/7 | Train Loss: 0.0123 | Train Accuracy: 89.7883 
          | Val Loss: 0.0027 | Val Accuracy: 90.2066

==============================
Epoch: 5/7 | Train Loss: 0.0097 | Train Accuracy: 92.215 
          | Val Loss: 0.0029 | Val Accuracy: 89.1736

==============================
Epoch: 6/7 | Train Loss: 0.0083 | Train Accuracy: 93.1538 
          | Val Loss: 0.0021 | Val Accuracy: 92.5207

==============================
Epoch: 7/7 | Train Loss: 0.0064 | Train Accuracy: 94.9872 
          | Val Loss: 0.004 | Val Accuracy: 86.281

==============================
Test Loss: 0.0032 | Test Accuracy: 88.0165


then it shows a graph of validatoin with trainning loss graph compared to epochs used, to see if there is an overfitting, 

and then you can choose an image to test on it the model

( note if you choose to use the pretrained model you will go directly to choose an image)
