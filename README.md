# FlowerClassifier
Create Own Flower Classifier from images by implementing an image classifier with TensorFlow
In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using this dataset from Oxford of 102 flower categories, you can see a few examples below.
The project is broken down into multiple steps:

- Load the image dataset and create a pipeline.
- Build and Train an image classifier on this dataset.
- Use your trained model to perform inference on flower images.

## Dataset :  [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102)
using tensorflow_datasets, we loaded the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). This dataset has 3 splits: 'train', 'test', and 'validation'.The training dataset is normalized and resized to 224x224 pixels as required by the pre-trained networks. The validation and testing sets are used to measure the model's performance on data it hasn't seen yet, and still need to normalize and resize the images to the appropriate size.

## Project_Image_Classifier_Project.ipynb 
- DataLoading: The Oxford Flowers 102 dataset is loaded using TensorFlow Datasets.
- Data Splits: The dataset is divided into a training set, a validation set, and a test set.
- Dataset Info: The number of examples in each set and the number classes in the dataset are extracted from the dataset info.
- Dataset Images: The shape of the first 3 images in the training set is printed using a `for` loop and the `take()` method.
- Plot Image: The first image from the training set is plotted with the title of the plot corresponding to the image label.
- Label mapping: The first image from the training set is plotted with the title of the plot corresponding to the class name using label mapping from the JSON file. *Note that the keys from class_names ranges from 1-102, while the index of the labels from model.predict() will range from 0-101. When mapping the labels to the names, the index should be +1 added in order to correctly map.*
- Data Normalization: The training, validation, and testing data is resized and normalized.
- Data Pipeline: A pipeline for each set is constructed with the necessary transformations
- Data Baching: The pipeline for each set return batches of images.
- Pre-trained Network: The pre-trained network, MobileNet, is loaded using TensorFlow Hub and its parameters are frozen.
- Feedforward Classifier: A new neural network is created using transfer learning. The number of neurons in the output layer correspond to the number of classes of the dataset.
- Training the Network: The model is configured for training using the `compile` method with parameters. The model is trained using the `fit` method and incorporating the validation set.
- Validation Loss and Accuracy: The loss and accuracy values achieved during training for the training and validation set are plotted using the `history` dictionary return by the `fit` method.
- Testing Accuracy: The network's accuracy is measured on the test data.
- Saving the Model: The trained model is saved as a Keras model (i.e. saved as an HDF5 file with extension `.h5`).
- Loading Model: The saved Keras model is loaded.
- Image Processing: The `process_image` function normalizes and resizes the input image. The image returned by the `process_image` function is a NumPy array with shape `(224, 224, 3)`.
- Inference: The `predict` function takes the path to an image and a saved model, and then returns the top K most probably classes for that image.
- Sanity Check: A `matplotlib` figure is created displaying an image and its associated top 5 most probable classes with actual flower names.

## Predict.py 
- Predicting Classes: The `predict.py` script reads in an image and a saved Keras model and then prints the most likely image class and it's associated probability.
- Top K Classes: The `predict.py` script allows users to print out the top K classes along with associated probabilities.
- Displaying Class Names: The `predict.py` script allows users to load a JSON file that maps the class values to other category names.
- Use [Command-Line Arguments Parser](https://www.tutorialspoint.com/python/python_command_line_arguments.htm)

### How to Run
predict.py file uses a trained network to predict the class for an input image. The predict.py module should predict the top flower names from an image along with their corresponding probabilities.

Basic usage:
> `$ python predict.py /path/to/image saved_model`

Options:
* --top_k : Return the top KK most likely classes:
> `$ python predict.py /path/to/image saved_model --top_k KK`

* --category_names : Path to a JSON file mapping labels to flower names:
> `$ python predict.py /path/to/image saved_model --category_names map.json`

The best way to get the command line input into the scripts is with the argparse module in the standard library. You can also find a nice tutorial for argparse here.

#### Examples
For the following examples, we assume we have a file called orchid.jpg in a folder named/test_images/ that contains the image of a flower. We also assume that we have a Keras model saved in a file named my_model.h5.

Basic usage:
> `$ python predict.py ./test_images/orchid.jpg my_model.h5`

Options:
Return the top 3 most likely classes:
> `$ python predict.py ./test_images/orchid.jpg my_model.h5 --top_k 3`

Use a `label_map.json` file to map labels to flower names:
> `$ python predict.py ./test_images/orchid.jpg my_model.h5 --category_names label_map.json`

### Testing Images
In the Command Line Interface workspace we have 4 images in the `./test_images/ folder` to check prediction.py module. The 4 images are:
* `cautleya_spicata.jpg`
* `hard-leaved_pocket_orchid.jpg`
* `orange_dahlia.jpg`
* `wild_pansy.jpg`

### Next step
- Use at least one form of regularization
- Try to get the difference between training and validation accuracy 
