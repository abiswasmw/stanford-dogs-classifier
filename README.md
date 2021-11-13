# Stanford Dogs Classifier
Identify dog breeds from images.

This project builds a classifier that can predict dog breeds from images. The classifier is trained on the [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs) TensorFlow dataset.

![Image](./resources/sample_predictions.png)

## Contents
1. [Intallation and Setup](#Intallation-and-Setup)
2. Run
3. Tutorial
   * Download and preprocess data
   * Create a machine learning model
   * Train the model
   * Validate and test the model

## Installation and Setup

First, clone this repository.
```
git clone https://github.com/aribiswas/stanford-dogs-classifier.git
```

The required packages for this project are specified in a **requirements.txt** file. You will need to create a virtual environment from the requirements file for this project. An easy way to do this is to download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html). 

Once installed, navigate to the cloned repository and execute the following command:
```
conda create --name <YOUR-ENV-NAME> --file requirements.txt
```
Activate the conda environment and you are ready to run the project.
```
conda activate <YOUR-ENV-NAME>
```

## Run

The entry point script for this project is **run.py**. You can use this script to identify dog breeds from image URLs.

From the project root execute the entry point script. This will prompt you to enter the URL of an image that you want to identify.
```
python run.py
Enter an url for the image, 0 to quit: https://media.nature.com/lw800/magazine-assets/d41586-020-03053-2/d41586-020-03053-2_18533904.jpg
```
Altenately, you can provide the URLs as command line arguments.
```
python run.py --url <url1> <url2> ...
```

## Tutorial

The objective of this project is to build a classifier model that can accurately predict dog breeds from images. I use TensorFlow 2.0 and the standford-dogs TensorFLow dataset to train the classifier. Following are the high level steps for reproducing my results:

### Download and preprocess data

The first step is to download the stanford-dogs dataset which contains images of 120 dog breeds from around the world. You can find more information [here](https://www.tensorflow.org/datasets/catalog/stanford_dogs). Data processing utilities are written in the **dataprocessor.py** module, but I will explain them in details.

Download the data using the tfds.load .
```
import tensorflow as tf
import tensorflow_datasets as tfds

(ds_train, ds_test), ds_info = tfds.load('stanford_dogs',
                                          split=['train', 'test'],
                                          shuffle_files=True,
                                          as_supervised=False,
                                          with_info=True,
                                          data_dir='data/tfds')
```



