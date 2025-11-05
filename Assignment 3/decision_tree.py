# -------------------------------------------------------------------------
# AUTHOR: Hye Won Kang
# FILENAME: decision_try.py
# SPECIFICATION: This program trains and evaluates a Decision Tree classifier to predict whether a taxpayer will cheat 
#                based on refund status, marital status, and taxable income. 
#                It reads three different training datasets and one test dataset, encodes categorical attributes (Refund and Marital Status), 
#                and computes the average prediction accuracy over 10 runs using the Gini index.
# FOR: CS 4440 (Data Mining) - Assignment #3
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
    data_training = np.array(df.values)[:,1:] #creating a training matrix without the id (NumPy library)

    #transform the original training features to numbers and add them to the 5D array X. For instance, Refund = 1, Single = 1, Divorced = 0, Married = 0,
    #Taxable Income = 125, so X = [[1, 1, 0, 0, 125], [2, 0, 1, 0, 100], ...]]. The feature Marital Status must be one-hot-encoded and Taxable Income must
    #be converted to a float.
    # X =

    for row in data_training:
        # Refund
        if row[0].strip().lower() == 'yes':
            refund = 1
        else:
            refund = 0

        # marital status - one hot encode
        maritalStatus = row[1].strip().lower()

        # initialization to 0
        single = 0
        divorced = 0
        married = 0

        if maritalStatus == 'single':
            single = 1
        elif maritalStatus == 'divorced':
            divorced = 1
        elif maritalStatus == 'married':
            married = 1

        # Taxable Income string to float
        income_str = row[2].strip().lower()
        if income_str.endswith('k'):
            income_str = income_str.replace('k', '')
        income = float(income_str)

        # add them to the 5D array X
        X.append([refund, single, divorced, married, income])

    X = np.array(X, dtype = float)
    #transform the original training classes to numbers and add them to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    #--> add your Python code here
    # Y =

    for row in data_training:
        # Cheat Class
        cheat = row[3].strip().lower()
        if cheat == 'yes':
            Y.append(1)
        else:
            Y.append(2)
            
    # add them to the vector Y
    Y = np.array(Y, dtype = int)

    sum_accuracies = 0.0

    #loop your training and test tasks 10 times here
    for i in range (10):

       #fitting the decision tree to the data by using Gini index and no max_depth
       clf = tree.DecisionTreeClassifier(criterion = 'gini', max_depth=None)
       clf = clf.fit(X, Y)

       #plotting the decision tree
       #tree.plot_tree(clf, feature_names=['Refund', 'Single', 'Divorced', 'Married', 'Taxable Income'], class_names=['Yes','No'], filled=True, rounded=True)
       #plt.show()

       #read the test data and add this data to data_test NumPy
       #--> add your Python code here
       # data_test =
       df_test = pd.read_csv('cheat_test.csv', sep=',', header=0)   #reading a dataset eliminating the header (Pandas library)
       data_test = np.array(df_test.values)[:,1:] #creating a training matrix without the id (NumPy library)

       # Accuracy count
       correct = 0
       total = 0
       
       for data in data_test:
           #transform the features of the test instances to numbers following the same strategy done during training, and then use the decision tree to make the class prediction. For instance:
           #class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0], where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           #--> add your Python code here

            # Refund
            if data[0].strip().lower() == 'yes':
                refund = 1
            else:
                refund = 0

            # marital status - one hot encode
            maritalStatus = data[1].strip().lower()

            # initialization to 0 
            single = 0
            divorced = 0
            married = 0

            if maritalStatus == 'single':
                single = 1
            elif maritalStatus == 'divorced':
                divorced = 1
            elif maritalStatus == 'married':
                married = 1
            
            # Taxable Income string to float
            income_str = data[2].strip().lower()
            if income_str.endswith('k'):
                income_str = income_str.replace('k', '')
            income = float(income_str)

            # class_predicted = clf.predict([[1, 0, 1, 0, 115]])[0] is used to get an integer [0] is the Cheat class
            class_predicted = int(clf.predict([[refund, single, divorced, married, income]])[0])
                                
            # Cheat Class
            cheat = data[3].strip().lower()
            if cheat == 'yes':
                true_label = 1
            else:
                true_label = 2
        
            #compare the prediction with the true label (located at data[3]) of the test instance to start calculating the model accuracy.
            #--> add your Python code here
            if class_predicted == true_label:
                correct += 1
            total += 1
        
       #find the average accuracy of this model during the 10 runs (training and test set)
       #--> add your Python code here

       accuracy = correct/ float(total)
       sum_accuracies += accuracy 
    ave_accuracy = sum_accuracies / 10

    #print the accuracy of this model during the 10 runs (training and test set).
    #your output should be something like that: final accuracy when training on cheat_training_1.csv: 0.2
    #--> add your Python code here

    print(f'final accuracy when training on {ds} {ave_accuracy: .2f}')



