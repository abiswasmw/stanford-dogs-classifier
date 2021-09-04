# Stanford Dogs Classifier
Identify dog breeds from images.

This project builds a classifier that can predict dog breeds from images. The classifier is trained on the [stanford_dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs) TensorFlow dataset.

![Image](./resources/sample_predictions.png)

## Setup
1. Clone this project to get all the necessary files.
2. Download [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage the project environment.
3. Open a terminal and cd into the project directory.
4. Create a Conda environment from the requirements.txt file.
```
conda create --name sdc --file requirements.txt
```
5. Activate the conda environment and you are ready to go.
```
conda activate sdc
```

## Run

The entry point script is run.py. You can use this script to identify dog breeds from images. The images can be provided as URLs.

1. From the project root execute the entry point script.
```
python run.py
```
This will prompt you to enter the url of a dog image that you want to identify. For example,
```
Enter an url for the image, 0 to quit: https://media.nature.com/lw800/magazine-assets/d41586-020-03053-2/d41586-020-03053-2_18533904.jpg
```
You can also enter multiple urls.

2. Or, provide urls as command line arguments.
```
python run.py --url <url1> <url2> ...
```



