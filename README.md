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

## Usage

### 1) Setup
Clone the Repository:
```bash
git clone https://github.com/satheeshbhukya/DeepLense-Task.git
cd DeepLense-Task
```
- For **Common Task**, use `deeplense1.ipynb` notebook.
- For **Lens Finding**, use `deeplense2.ipynb` notebook.

### 2) Hyperparameters Setting
Modify the **CFG** class to change hyperparameters:
```python
class CFG:
    lr = 1e-4
    batch_size = 32 
    num_classes = 2
    size = [224, 224]  
    nfold = 10
    custom_model = False
    model_name = "tf_efficientnet_b4_ns"  
    target_col = "target"
    epochs = 10  
    seed = 42
    transform = False
    weight_decay = 1e-5
    num_workers = 2
    train = True
    debug = False
    metric_type = "roc_auc"
    scheduler_type = "CosineLRScheduler"
    optimizer_type = "AdamW" 
    loss_type = "BCEWithLogitsLoss"
    is_cross_validate = False
    max_grad_norm = 1000
    lr_max = 2e-4
    epochs_warmup = 1.0
    pos_weight = 5
    meta_count = 3
    device = device 
```

### 3) Augmentation
Use `get_transforms` function for data augmentation:
```python

def get_transforms(*, data):
    if data == 'train':
        return A.Compose([
            A.Resize(CFG.size[0], CFG.size[1]),
            A.OneOf([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.5),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
                A.GaussianBlur(blur_limit=(3, 7), p=0.5),
            ], p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    elif data == 'test':
        return A.Compose([
            A.Resize(CFG.size[0], CFG.size[1]),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
```

### 4) Training and Evaluation
Use the `Train` class to train and evaluate the model:
```python
def main():
    if CFG.train: 
        # Train
        train = Train(CFG)
        df = train.train_loop(train_df, val_df)
    return df 

if __name__ == '__main__':
    pred_df = main()
```

## Dependencies
To run the notebooks, install the following dependencies:
```bash
pip install torch torchvision timm tensorflow keras numpy matplotlib albumentations scikit-learn
```
---
**Contact:** [satheeshbhukyaa@gmail.com]
