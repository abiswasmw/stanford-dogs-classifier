# Stanford Dogs Classifier
Identify dog breeds from images.

This project builds a classifier that can predict dog breeds from images. The classifier is trained on the [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs) TensorFlow dataset.

![Image](./resources/sample_predictions.png)

## Contents
1. [Intallation and Setup](#Intallation-and-Setup)
2. Run
3. Tutorial
   * Download and preprocess dataset
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

### Download and preprocess dataset

The stanford-dogs dataset contains images of 120 dog breeds from around the world. The first step is to download the dataset in your project folder. The **dataprocessor.py** module contains the required utilities to download and preprocess the images. I will explain the high level steps in details.

Write a function to download the dataset. I use the tfds.load function which downloads the dataset to a folder of your choice (data/tfds in the code).
```
import tensorflow as tf
import tensorflow_datasets as tfds

def load_dataset():
	(ds_train, ds_test), ds_info = tfds.load(
                                           'stanford_dogs', 
                                            split=['train', 'test'],
                                            shuffle_files=True,
                                            as_supervised=False,
                                            with_info=True,
                                            data_dir='data/tfds'
                                          )
```
Use the show_examples() function in dataprocessor.py to view a few sample images from the dataset.
```
import dataprocessor as proc

proc.load_dataset()
proc.show_examples()
```
The images in the dataset have different sizes. In order to train a model from these images, you need to resize the images to a reasonable size. First, use the analyze function to plot a histogram of image widths and heights. This gives a rough sense of the median image dimensions.
```
proc.analyze()
```
![Analyze_Image](./resources/analyze.png)

Next, write a function to process an (image,label) tuple from the dataset. In this function, I cast the input image to float, resize the dimensions, normalize the resultant image and one hot encode the label.
```
def preprocess(data):
    image_size = (224, 224)
    num_labels = 120
    processed_image = data['image']
    label = data['label']
    processed_image = tf.cast(processed_image, tf.float32)
    processed_image = tf.image.resize(processed_image, image_size, method='nearest')
    processed_image = processed_image / 255.
    label = tf.one_hot(label, num_labels)
    return processed_image, label
```

Now prepare an input pipeline for the dataset. I use the map function to preprocess each entry from the dataset.
```
def prepare(dataset):
    dataset = dataset.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.cache()
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset
```

