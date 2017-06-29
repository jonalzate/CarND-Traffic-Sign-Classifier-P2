#**TRAFFIC SIGN RECOGNITION PROJECT**

---

####Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

####Dataset Exploration

** Dataset Summary **

I used the Python pre-built functions to calculate summary statistics of the traffic signs data set like len() and shape:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

**Exploratory Visualization**

The first step is to see what is in the data set and pair labels indexes 
with the corresponding names from the csv file and store them in a dictionary.
Next display a couple of signs with their names

![image1](/screenshots/signsloaded.png)

The bar chart below displays the data unevenly distributed where some classes have more data samples than others, this can have an impact in the training and results from the network 

![image2](/screenshots/signclassesplot.png)

####Design and Test Model Architecture

**Data Preprocessing**

* Grayscale convertion - Taken as advice from the sign classification article by Sermanet and LeCun, images were converted to grayscale to reduce training time. Images in grayscale tend to perform better in training for different lightning conditions.
* Normalize images - Done to keep computed values with a mean = 0 and equal variance 

Here is an output of the images once preprocessed

![image3](/screenshots/preprocessedData.png)

**Model Architecture**

The model is the one developed in class, named LeNet, which accepts a 32x32x1, where 1 is the number of color channels. We use one channel because we are taking images in grayscale as input. LeNet consist of the following architecture:

* Layer 1: Convolutional Outputs 28x28x6
 * Activation: ReLu activation function
 * Pooling: Max pooling with output 14x14x6

* Layer 2: convolutional outputs 10x10x6
 * Activation: ReLu activation function
 * Pooling: Max pooling with output 14x14x6
 * Flatten: Flattened to output 400

* Layer 3: Fully connected layer output 120 
 * Activation: ReLu activation function

* Layer 4: Fully connected layer output 84 
 * Activation: ReLu activation function

* Layer 5: Fully connected layer output 43 

**Model Training**
For training the following setup was used:

* BATCH_SIZE=120
* EPOCHS=15
* learn_rate=0.009
* AdamOptimizer instead of GradientDescent

**Solution Approach**
By tuning the hyperparameters for training, I was able to gett the following results after training for 15 epochs:

* Training Accuracy = 0.980
* Validation Accuracy = 0.967
* test accuracy = 0.896

####Test Model on New Images

**Acquiring New Images**

After a brief web search the five images were selected. They had different resolutions and a few were from the classes that had the least samples in the dataset. I noticed after preprocessing the training images they looked blurry and it was difficult to tell what sign it was. This could lead to believe that resizing from large resolutions can have its effects on the performance of the model.
Below are the loaded test images and after preprocessing.

![image4](/screenshots/testimages.png)

![image5](/screenshots/preprocessedTestImage1.png)

![image6](/screenshots/preprocessedTestImage2.png)

**Performance on New Images**

Running the model on the test images downloaded from the internet did not yield the expected results even after achieving a validation accuracy of 96.7% the accuracy on the test set was a mere 20%. The model only identified one of the test images as correct

![image7](/screenshots/prediction1.png)

![image8](/screenshots/prediction2.png)

![image9](/screenshots/prediction3.png)

![image10](/screenshots/prediction4.png)

![image11](/screenshots/prediction5.png)



**Model Certainty - Softmax probabilities**

The top\_k method from tensorflow was used to compute the top 3 probabilities that the model calculated as the correct for each test image.

![image12](/screenshots/topPredictions.png)
