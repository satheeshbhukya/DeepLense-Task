## ML4SCI GSOC DeepLense Task 
This GitHub repository contains two Folders, each of which focuses on a different deep learning task. 
## Common Task: Multi-class Classification 
The notebook in Common Task demonstrates a simple image classification task using a convolutional neural network (Transfer Learning). 
The dataset used in this notebook is the one provided for common task, which is a collection of strong lensing image. 
The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance. 
# Results on various Models:  

| Model    | Epochs | Batch Size | Learning Rate | Roc_Auc |
|----------|--------|------------|---------------|----------|
| ResNet18 | 5     | 64         | 0.0004         | 0.97    |
| EfficientNetB4 | 10 | 64        | 0.0002        | 0.98    |
| Ensemble  | -    | -         |                  | 0.98   |

