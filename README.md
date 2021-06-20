# Exercise-Deep_Learning_Malaria_Dataset

## Aim 
Exploring ways to implement machine learning CNN model on small datasets and to obtain an effective model, by applying Image Data Generator and Transfer Learning (VGG16). <br>

## Backgroud
#### About the dataset 
The Malaria dataset is frequently used as a practice on implement machine learning model. Malaria is an infectious disease casused by Plasmodium parasites, easily transmitted through mosquito bites. Many fatal cases was happened in Africa due to the poor access to healthcare resources for malaria screenings. To reduce the burden for microscopists in resource-constrained regions and improve diagnostic accuracy, researchers at the Lister Hill National Center for Biomedical Communications (LHNCBC), part of National Library of Medicine (NLM), have developed a mobile application that runs on a standard Android smartphone attached to a conventional light microscope. <br>

The Malaria dataset contains a total number of 27,558 blood cell images and annotated by expert as infected and uninfected cells. CNN model was commonly choose as it can identify the feature of infected and uninfected cells. However, it tooks around two hours to complete the model training for this simple classification due to the size of the datasets. <br>

#### About this repository
In real-world setting, it is more often that we may have a small dataset to work with. <br>
Therefore, I would like to reduce the dataset to 500 images and explore the below two methods that commonly used to improve model performance for small dataset. <br>
Then, exploring the effect of below ways in improving the model.
1. Image Data Generator
2. Transfer Learning (VGG16)

## Exploring the dataset 
The repository consist of two folders. There were both 13779 images for 'infected' and 'uninfected' blood cells. 
#### Infected cells sample
<img src="image/infected01.png" width="100">
<img src="image/infected02.png" width="100">
<img src="image/infected03.png" width="100">
<img src="image/infected04.png" width="100">
<img src="image/infected05.png" width="100">

#### Uninfected cells sample
<img src="image/Uninfected01.png" width="100">
<img src="image/Uninfected02.png" width="100">
<img src="image/Uninfected03.png" width="100">
<img src="image/Uninfected04.png" width="100">
<img src="image/Uninfected05.png" width="100">

## Import images
Images are imported in 64x64 resolution. 250 images are randomly selected from each class and the infected cell images labeled as Class1 while uninfected cell images are Class2.<br>
<img src="image/importimage.png" width="600">

The dataset is split to training set and testing set in 80:20 ratio respectively. <br>

## Basic Convolutional Neural Networks (CNN) 
The basic CNN model consist of three parts.<br>
The first part is a feature extraction part which consist of three convolution layers and two Max Pooling layers.<br>
It follows with a flattening part which can make the model more generalize and prevent over-fitting.<br>
The models end with three layers of neuronn network for classification and the activation function 'sigmoid' will predict the image pass through is an infected (as 1) or uninfected (as 0) cell image.<br>

The below is the code and the structure of the model.<br>
<img src="image/model_CNN.png" width="600">
<img src="image/visualize_model.png" width="600">

The basic model trained directly using the 500 images, with batch_size = 20 and epochs = 20.<br>

Since the dataset is small and the model is relatively simple, the model is overfitted and the accuracy on the testing set is only 0.6. The below graph is to visualize the training performance. <br>
<img src="image/model1_performance.png" width="600">

The below methods applied on the basic model, aimed at improving the model performance and two new models created.<br>
Model 1. Image Data Generator
Model 2. Transfer Learning (VGG16)


## Model 1: Apply ImageDataGenerator
Model 1 is applying ImageDataFenerator on the dataset which create random variation on the images. Random rotation, zoom, shift, shear or flip applied on different images and pass througgh the basic CNN model structure. <br>
<img src="image/model_CNNimagedatagenerator.png" width="600">

ImageDataFenerator improve the model significantly and result for a more generalized model. Both the training accuracy and actual accuracy are reaching simliar accuracy at over 90% and is not overfitting to the training data. <br>
<img src="image/model2_performance.png" width="600">

## Model 2: Apply transfer leaning (VGG16)
Transfer learning is another method that commonly used in machine learning for prediction model where the medol will reused some parts of a pre-trained model as a starting point of the new moel training. <br>

VGG16 is a CNN model 

<img src="image/model_transferlearning.png" width="600">