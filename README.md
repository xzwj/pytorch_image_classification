# PyTorch Implementations for Image Classification
## Datasets
- MNIST
- CIFAR10

## Model
- soft-max regression (SR)
- MLP
- LeNet5
- VGG11/13/16/19

## Experiment Settings
<table>
    <caption>Table 1: Hyperparameter settings. lr—learning rate, bs—batch size, optim—optimizer, mom—momentum, wd—weight decay, num_epoch—total number of epochs. Adjust schedule of VGG16 learning rate: begin with 0.1, decays by 0.1 every 20 epochs.</caption>
    <tr>
        <td></td>
        <td colspan="3">MNIST</td>
        <td colspan="4">CIFAR10</td>
    </tr>
    <tr>
        <td></td>
        <td>SR</td>
        <td>MLP</td>
        <td>LeNet5</td>
        <td>SR</td>
        <td>MLP</td>
        <td>LeNet5</td>
        <td>VGG16</td>
    </tr>
    <tr>
        <td>layer size</td>
        <td>512</td>
        <td>(512, 512)</td>
        <td>-</td>
        <td>512</td>
        <td>(512, 512)</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>lr</td>
        <td>1e-3</td>
        <td>1e-3</td>
        <td>1e-3</td>
        <td>1e-3</td>
        <td>1e-3</td>
        <td>1e-3</td>
        <td>Adjust</td>
    </tr>
    <tr>
        <td>bs</td>
        <td>32</td>
        <td>32</td>
        <td>32</td>
        <td>32</td>
        <td>32</td>
        <td>32</td>
        <td>32</td>
    </tr>
    <tr>
        <td>optim</td>
        <td>Adam</td>
        <td>Adam</td>
        <td>Adam</td>
        <td>SGD</td>
        <td>SGD</td>
        <td>SGD</td>
        <td>SGD</td>
    </tr>
    <tr>
        <td>mom</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>0.9</td>
        <td>0.9</td>
        <td>0.9</td>
        <td>0.9</td>
    </tr>
    <tr>
        <td>wd</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>-</td>
        <td>5e-4</td>
    </tr>
    <tr>
        <td>dropout</td>
        <td>-</td>
        <td>0.2</td>
        <td>-</td>
        <td>-</td>
        <td>0.2</td>
        <td>-</td>
        <td>-</td>
    </tr>
    <tr>
        <td>num_epoch</td>
        <td>20</td>
        <td>30</td>
        <td>30</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
        <td>100</td>
    </tr>
</table>

## Results
<table>
    <caption>Table 2: Experiment results. “Loss” means average loss of batches (each batch has 32 samples).</caption>
    <tr>
        <td colspan="3">MNIST</td>
        <td colspan="3">CIFAR10</td>
    </tr>
    <tr>
        <td>Model</td>
        <td>Acc</td>
        <td>Loss</td>
        <td>Model</td>
        <td>Acc</td>
        <td>Loss</td>
    </tr>
    <tr>
        <td>SR</td>
        <td>92.9</td>
        <td>0.263</td>
        <td>SR</td>
        <td>31.4</td>
        <td>3.491</td>
    </tr>
    <tr>
        <td>MLP</td>
        <td>98.6</td>
        <td>0.094</td>
        <td>MLP</td>
        <td>56.9</td>
        <td>1.273</td>
    </tr>
    <tr>
        <td>LeNet5</td>
        <td>99.1</td>
        <td>0.045</td>
        <td>LeNet5</td>
        <td>71.8</td>
        <td>0.801</td>
    </tr>
    <tr>
        <td>VGG16</td>
        <td>-</td>
        <td>-</td>
        <td>VGG16</td>
        <td>92.0</td>
        <td>0.318</td>
    </tr>
</table>


## Code Layout
```
.
├── experiments/
├── data/
│   ├── cifar-10-batches-py/
│   └── MNIST/
├── models/
│   ├── data_loaders.py
│   ├── nets.py
│   └── vgg.py
├── requirements.txt
├── train.py
└── utils.py
```

-	train.py: contains main training loop
-	utils.py: utility functions
-	evaluate.py: contains main evaluation loop
-	data/: store datasets
-	models/data_loaders.py: data loaders for each dataset
-	models/vgg.py: VGG11/13/16/19
-	models/nets.py: Soft-max Regression, MLP, LeNet5 and evaluation metrics
-	experiments/: store hyperparameters, model weight parameters, checkpoint and training log of each experiments

## Requirements
Create a conda environment and install requirements using pip:
```
>>> conda create -n ass2 python=3.7
>>> source activate ass2
>>> pip install -r requirements.txt
```
## How to Run
Train a model with the specified hyperparameters:
```
>>> python train.py --model {model name} --dataset {dataset name} --model_dir {hyperparameter directory}
```
For example, using VGG16 and CIFAR10 to train a model with the hyperparameters in experiments/cifar10_vgg16/params.json:
```
>>> python train.py --model vgg16 --dataset cifar10 --model_dir experiments/cifar10_vgg16
```
It will automatically download the dataset and puts it in “data” directory if the dataset is not downloaded. During the training loop, best model weight parameters, last model weight parameters, checkpoint, and training log will be saved in experiments/cifar10_vgg16.
