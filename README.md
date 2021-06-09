
# InFlow: Robust outlier detection utilizing Normalizing Flows

We provide the code for training and evaluating the InFlow model on CIFAR 10 dataset. 

## Requirements

To run our experimental code, you need the following python packages:

* PyTorch
* numpy 
* torchvision
* alibi-detect
* ood-metrics
* sklearn


All the above packages can be installed using ```pip``` command. To set up the environment, you need to download the publicly available torchvision datasets such as MNIST, FashionMNIST, SVHN, CIFAR-10 etc as image files and keep each of the datasets in separate folders of the project directory. 

## Training

To train the InFlow model once the dataset preparation is complete, run the following command in the project directory:

```train
python train.py 
```
Note: You need to change the path with respect to the folder where datasets are placed in the project directory

## Evaluation

To evaluate the trained InFlow model on several datasets, run the following command:

```eval
python eval.py 
```

## Pre-trained Models
You can also skip training the model and directly run the evaluation by using the provided pre-trained model. The pretrained model can be found in the ``` /ckptdir ``` directory.. 
