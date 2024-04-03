## **Classification of barbell exercises using accelerometer and gyroscope data**

## **Problem Statement**

The goal of the project is to analyse wristband accelerometer and gyroscope data obtained during strength training sessions, to build models that can classify exercises and count repititions. This problem is inspired from the paper titled ‘Exploring the Possibilities of Context-Aware Applications for Strength Training’ by Dave Ebbelar.  

## **Data Preparation**
The collected data is of 5 exercises namely Bench Press, Deadlift, Overhead Press, Row, and Squat. The data exported from the accelerometer and gyroscope sensors are in csv format. The representative representative dataset is collected from 5 participants performing barbell exercises in 3 sets of 5 repetitions and 10 repititions. Thus there are a total of 150 sets of exercise data and 37 sets of rest data( to examine the state change from rest to exercise) are included. These 187 csv files are a mix of both gyroscope and accelerometer data. Also, the labels have to be extracted from the name of csv file. Hence the data is not highly structured. 

## **Converting raw data**
The raw dataset contained a total of 69677 entries, each containing an epoch timestamp and x, y, and z-values from the sensor. The raw data is resampled in an interval of 200ms. This resampling ensures the dataset is not too big enough but at the same time has enough data.

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/84e70c34-9f94-41fb-b4f2-8860d266eb96" alt="Description of the image">
</p>
<p align="center">
  Accelerometer Data
</p>

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/73d00e21-f74a-4538-b21c-884c0522c755" alt="Description of the image">
</p>
<p align="center">
  Medium and heavy weight squats
</p>

The goal is to transform the data in a way that subtle noise is filtered and the parts of our data that explain most of the variance are identified. Two approaches are implemented: Low-pass Filtering, which can be applied to individual attributes, and Principal Component Analysis, which works across the entire dataset.   

## Low-pass Filtering
The Butterworth low-pass  filter can remove some of the high frequency noise in the dataset that might disturb the learning process.

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/6c86e5db-fd4a-4373-9a10-fcec540d305e">
</p>

## Principal Component Analysis

A principal component analysis (PCA) was conducted to find the features that could explain most of the variance. PCA was applied to all features excluding the target columns. The results are visualized in the below figure which shows that the explained variance drastically decreases after 3 components. Therefore, 3 components are selected and their values are included into the dataset. These 3 components together explain around 95% of the variance in the data

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/15a47846-9abd-468a-9413-15fd49582134">
</p>

## **Feature Engineering**
In feature engineering additional features like aggregated features, temporal abstraction features, frequency domain features,clustering were derived from the exisiting features. For temporal abstraction, rolling window average method is used with a window size of 5. This resulted in highly correlated attributes which can potentially lead to overfitting. Inorder to avoid this, 50% overlap between rows were allowed and the rest of the rows were removed.

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/c7dbf1fb-a3ad-40b9-825b-162e85592ddf" alt="Description of the image">
</p>
<p align="center">
  Original Signal
</p>

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/cc6237cb-156e-4e60-9b1c-ec1dec551852" alt="Description of the image">
</p>
<p align="center">
  Temporal abstraction ouput
</p>

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/c304bb29-d1b2-4fc5-812d-06228ab1272b" alt="Description of the image">
</p>
<p align="center">
  Frequency abstraction ouput
</p>

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/7d2a0273-71f7-47a0-8fca-ce38ba7610ab" alt="Description of the image">
</p>
<p align="center">
  Clusters
</p>


## **Modelling**
The dataset is now processed and ready for training. The dataset contains the 6 basic features, 2 scalar magnitude features, 3 PCA features, 16 time features, 88 frequency features and 1 cluster feature, summing upto 116 columns. The train test split graph is given below

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/72d6fd3d-4c67-4f72-bc36-4f64a5fb2763" alt="Description of the image">
</p>
<p align="center">
  Train Test Split Graph 
</p>


Split feature subsets : A total of 116 features were split into different sets namely:

feature_set_1 : basic features 
feature_set_2 : basic features + square features + pca features
feature_set_3: feature_set_2 + time features
feature_set_4: feature_set_3 + frequency features + cluster features

Feature selection : Forward feature selection was used to investigate which features contribute the most to performance as useless features could impact the performance of the algorithms. Using a simple decision tree, and gradually adding the best features, the results showed us that after 4 features the performance no longer significantly improved . The best features were stored ina variable named ‘selected features’.


<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/ab0048bb-f24c-4ccb-9668-380e5e38d3ef" alt="Description of the image">
</p>
<p align="center">
  Results of forward feature selection 
</p>

Models : First, an initial test runusing GridSearchCV was done to determine the performance of a selection of models and features. This test included the following models: Neural network, Random Forest, Support Vector Machine, K-nearest Neighbours, Decision Tree, Naive Bayes. Grid search was performed on all of the models.

## **Results**
The results per model can be seen below

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/3ac6fe0c-9de6-4bbc-a3b9-6d7e3b664a0f" alt="Description of the image">
</p>
<p align="center">
  Performance of all the models
</p>

Random Forest and Neural Network gave best accuracy with feature_set_4. Hence it can be seen that frequency components added to the increased accuracy. Random Forest was further optimized to evaluate the results. Random forest acquired an accuracy of 99% on test data. The confusion matrix is plotted below

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/16afc651-4acd-4149-8b60-236b93f58c8c" alt="Description of the image">
</p>
<p align="center">
  Random Forest confusion matrix
</p>

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/35429a96-47dd-431a-bb59-c909c86f64e7" alt="Description of the image">
</p>
<p align="center">
  Random Forest Classification report
</p>

Thus Random Forest model could correctly classify the  barbell excercises into its respective classes. 
Another aspect that was explored was counting the repititions of an exercise. To count repetitions, a simple peak counting algorithm was applied to the scalar magnitude acceleration data. To make sure small local peaks were neglected, a strong low-pass filter with a cutoff at 0.4 Hz was applied first. It was found that this method of counting repetitions has to be adjusted to the individual exercises for the best performance.   An overall error rate for counting repetitions was about 1% for the collected dataset. An example of 5 benchpress repetitions is shown below

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/b400ccdf-8fff-4f5f-bdf3-02bb761b8d3c">
</p>

<p align="center">
  <img src="https://github.com/saras56/Fitness_tracker_using_sensordata/assets/115695360/eaddd8b7-ec6b-466d-909e-1d5e66dfc0ad">
</p>
<p align="center">
  Actual repititions and predicted repititions for each exercise
</p>
