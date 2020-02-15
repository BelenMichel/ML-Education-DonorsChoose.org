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

The database is at a level projects. The target projects are those that have more than 60 days between the date the project was posted ("date_posted") and the date the project was fully funded ("datefullyfunded"). The data spans Jan 1, 2012 to Dec 31, 2013, thus we built three samples of data where our training sets start on the 1/1/12 and end 60 days before the testing set starts, allowing for the projects to be funded within those days. 

### 3. Results3

The repository also contains a results folder:

1. \results: this file has a csv file with the results for each of themetrics at each od the models trained. 

The Machin Learning models used were the following:  
1. Logistic Regression, 
2. Decision Tree, 
2. K-Nearest-Neighbors, 
2. Support Vector Machine, 
2. Random Forest Regressor, 
2. Ada Boost, 
2. Bagging.

Moreover, these models where tested with the following metrics: 
1. AUC_ROC, 
2. accuracy, 
3. precision, 
4. recall at the following thresholds 1%, 2%, 5%, 10%, 20%, 30%, 50%. 
