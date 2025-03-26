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
| tf_efficientnet_b4_ns | 10 | 64        | 0.0002        | 0.98    |
| Ensemble  | -    | -         |      -            | 0.98   |

## Specific Task 2 : Lens Finding 
The LensFinding folder has notebook focuses on a more specialized deep learning task, namely identifying gravitational lenses in astronomical images. Detecting gravitational lenses is important for understanding the structure and distribution of matter in the universe. The notebook includes all the necessary code to load the dataset, preprocess the images, define the CNN model, train the model, and evaluate its performance.

# Results on various Models: 

| Model    | Epochs | Batch Size | Learning Rate | Roc_Auc |
|----------|--------|------------|---------------|----------|
| ResNet18 | 20    | 64         | 0.0004         | 0.82    |
| tf_efficientnet_b2_ns | 10 | 64        | 0.0002        | 0.83    |
|tf_efficientnet_b4_ns  | 10    | 32         | 0.0004     | 0.85   | 
| Ensemble  | -    | -         |          -        | 0.85   |
