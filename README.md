# Machine Learning Pipeline - Education Data

This project explore some Machine Learning algorithms on a dataset of schools and education:

### 1. Link to the code

This code can also be find on the following repository:

https://github.com/BelenMichel/machinelearning.git

It can be run from the console by executing:
```bash
“python main.py”
```

The code files are named as follow:

1. "pipeline.py": it has all the functions to read a csv file, do the cleaning, split into different samples, split into training and testing data, create features, train different ML models and test them with different metrics. 

1. "main.py": it has a main function to train the desired ML models and test them with different metrics. This function takes csv files with the data and returns a csv file with the results of the defined metrics for each model. This file also contains the constants I defined to this specific exercise (such as the variables to use from the data, the models to train and the metrics to measure)

### 2. Write up 

The repository also contains a document called "Write_up" where the results of the trained models were explained.  

### 3. Data and Results

The repository also contains two folders:

1. \data: contains the original csv file with the raw data and six  additional csv files with the training and testing data for the three samples. These six csv files are already cleaned and the features and labels where created and saved in them. 
1. \results: this file has a csv file with the results for each of the models trained. 
