# Machine Learning Pipeline - DonorsChoose.org - Unfunded School Projects

This project explore some Machine Learning algorithms on a dataset of schools and education:

### 1. Code

This code can also be find on the following repository:

https://github.com/BelenMichel/ML-Education-DonorsChoose.org

It can be run from the console by executing:
```bash
“python main.py”
```

The code files are named as follow:

1. "pipeline.py": it has all the functions to read a csv file, do the cleaning, split into different samples, split into training and testing data, create features, train different ML models and test them with different metrics. 

1. "main.py": it has a main function to train the desired ML models and test them with different metrics. This function takes csv files with the data and returns a csv file with the results of the defined metrics for each model. This file also contains the constants I defined to this specific exercise (such as the variables to use from the data, the models to train and the metrics to measure)

### 2. Data 

The repository also contains a data folder:

1. \data: contains the original csv file with the raw data and six  additional csv files with the training and testing data for the three samples. These six csv files are already cleaned and they contain the features and labels created. 

The database is at a level projects, i.e. each row is a different project posted. We identified our target projects as those that have more than 60 days between the date the project was posted ("date_posted") and the date the project was fully funded ¬("datefullyfunded"). We wanted to predict the projects that were under-funded. Those represent an approximately a 30% of all the projects. With this information we know that the baseline accuracy of our prediction model should be around 70%. 

The data spans Jan 1, 2012 to Dec 31, 2013, thus we built three samples of data where our training sets start on the 1/1/12 and end 60 days before the testing set starts, allowing for the projects to be funded within those days. Moreover, we produced these samples twice, one with test sets extending over six month and another with testing sets extending four month -to account for cases where we would not know the funding’s that occur after Dec 31, 2013. 

### 3. Results

The repository also contains a results folder:

1. \results: this file has a csv file with the results for each of themetrics at each od the models trained. 

The Machin Learning models used were the following:  
a) Logistic Regression, 
b) Decision Tree, 
c) K-Nearest-Neighbors, 
d) Support Vector Machine, 
e) Random Forest Regressor, 
f) Ada Boost, 
g) Bagging.

Moreover, these models where tested with the following metrics: 
a) AUC_ROC, 
b) accuracy, 
c) precision, 
d) recall at the following thresholds 1%, 2%, 5%, 10%, 20%, 30%, 50%. 



