import re
import ast
import time
import itertools
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import pyspark
from pyspark.ml.feature import Imputer
import math
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors
from numpy import allclose
from pyspark.sql import Row
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, VectorSlicer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession

sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)



def parse_raw_row(row):
    '''
    for each row in the raw data,  output is a list of label and all the features:
        - [label, feature_1, feature_1, ...]
    For first 13 features, change the data type to number.
    Remaining features will of type string.
    For null values, populate None
    '''

    row_values = row.split('\t')
    for i, value in enumerate(row_values):
        if i <14:
            row_values[i] = float(value) if value != '' else None
        else:
            row_values[i] = value if value != '' else None
    return row_values

def imputeNumeric(numeric_DF, endCol):
    '''
    takes a spark df with continuous numeric columns
    outputs a spark df where all null values are replaced with the column average

    the first column, which is the outcome values, are preserved
    '''
    outputColumns=["{}_imputed".format(c) for c in numeric_DF.columns[1:endCol]]
    catColumns = ["{}".format(c) for c in numeric_DF.columns[endCol:]]

    imputer = Imputer(
        inputCols=numeric_DF.columns[1:endCol],
        outputCols=outputColumns
        )

    model = imputer.fit(numeric_DF)

    imputedDF = model.transform(numeric_DF).select(['_1']+outputColumns+catColumns)

    return imputedDF

def scaleFeatures(imputedDF, endCol):
    '''
    inputs imputed data frame with no null values and continuous features
    transforms the data frame into 2 column data frame with first column as label and second column as dense vector of features
    scales all features using the StandardScalar
    returns 2 column dataframe with scaled features
    '''

    transformedImputedDF = imputedDF.rdd.map(lambda x: (x[0], Vectors.dense(x[1:endCol]))).toDF(['label', 'x'])

    scaler = StandardScaler(inputCol="x",
                            outputCol="features",
                            withStd=True, withMean=True)

    scalerModel = scaler.fit(transformedImputedDF)
    scaledDF = scalerModel.transform(transformedImputedDF).select(['label', 'features'])

    return scaledDF


def processNumeric(dataRDD):
    '''takes input rdd and extracts only numeric columns
    then processes the numeric columns with imputation for null values and standard scaling
    outputs RDD ready for gradient descent algorithm evaluation'''

    dataDF = dataRDD.map(lambda x: x.split('\t'))\
            .map(lambda x: x[:14])\
            .map(lambda x: list(map(lambda y: float(y) if y!='' else None, x)))\
            .toDF()

    #drop features with the most missing values
    dataDF = dataDF.drop('_2','_11','_13')


    imputedDataDF = imputeNumeric(dataDF, 11)
    scaledDataDF =  scaleFeatures(imputedDataDF, 11)

    return scaledDataDF.rdd


############################################## READ DATA HERE ############################################
#rawTrainRDD = sc.parallelize(ast.literal_eval(open("eda.txt", "r").read()))
rawTrainRDD = sc.textFile('gs://w261-tktruong/data/train.txt')

otherTrain, partialTrain = rawTrainRDD.randomSplit([0.8, 0.2], seed = 2018)

rtrainRDD, rtestRDD = partialTrain.randomSplit([0.8, 0.2], seed = 2018)

trainRDD = processNumeric(rtrainRDD)
testRDD = processNumeric(rtestRDD)
###########################################################################################################


#ALL ALGORITHM FUNCTIONS

def sigmoid(arg):
    """helper function used to prevent math range error

    also makes loss and gradient functions easier to read"""

    if arg < 0:
        return 1 - 1 / (1 + math.exp(arg))
    else:
        return 1 / (1 + math.exp(-arg))

def LRLoss(augmentedData, W):
    """Takes augmented data and calculates log loss for current model vector W

    AUGUMENTATION MUST OCCUR BEFORE INPUT TO FUNCTION

    data must be transformed in the following way to use this function:
        y = 1 --> y' = 1
        y = 0 --> y' = -1"""

    loss = augmentedData.map(lambda x: -math.log(1+sigmoid(x[0]*np.dot(W, x[1]))))\
                        .sum()
    return loss

def GDUpdate(augmentedData, W, learningRate = 0.05, regType = None, regParam = 0.1):

    """Takes augmented data and calculates each gradient step to current model vector W

    AUGUMENTATION MUST OCCUR BEFORE INPUT TO FUNCTION

    data must be transformed in the following way to use this function:
        y = 1 --> y' = 1
        y = 0 --> y' = -1

    Includes the regularization terms for:
        1. L1 - lasso
        2. L2 - Ridge"""

    new_model = None

    if regType == 'ridge':
        L2reg = W*1
        L2reg[0] = 0 #first value is the y-intercept (bias) term and should be removed from regularization

        grad = augmentedData.map(lambda x: -x[0]*(1-sigmoid(x[0]*np.dot(W, x[1])))*x[1]+2*regParam*L2reg)\
                            .reduce(lambda x,y: x + y)

    elif regType == 'lasso':
        L1reg = W*1
        L1reg[0] = 0 #first value is the y-intercept (bias) term and should be removed from regularization
        L1reg = (L1reg>0).astype(int)*2-1

        grad = augmentedData.map(lambda x: -x[0]*(1-sigmoid(x[0]*np.dot(W, x[1])))*x[1]+regParam*L1reg)\
                            .reduce(lambda x,y: x + y)

    else:
        grad = augmentedData.map(lambda x: -x[0]*(1-sigmoid(x[0]*np.dot(W, x[1])))*x[1])\
                            .reduce(lambda x,y: x + y)

    new_model = W-learningRate*grad

    return new_model

def GradientDescent(trainRDD, wInit, testRDD = None, nSteps = 20,
                    learningRate = 0.025, regType = None, regParam = 0.1, verbose = False):
    """
    Perform nSteps iterations of OLS gradient descent and
    track loss on a test and train set. Return lists of
    test/train loss and the models themselves.
    """
    # initialize lists to track model performance
    train_history, test_history, model_history = [], [], []

    # perform n updates & compute test and train loss after each
    model = wInit
    for idx in range(nSteps):

        model = GDUpdate(trainRDD, model,learningRate, regType, regParam)
        training_loss = LRLoss(trainRDD, model)
        train_history.append(training_loss)
        if testRDD != None:
            test_loss = LRLoss(testRDD, model)
            test_history.append(test_loss)

        model_history.append(model)

        # console output if desired
        if verbose:
            print("----------")
            print(f"STEP: {idx+1}")
            print(f"training loss: {training_loss}")
            if testRDD != None:
                print(f"test loss: {test_loss}")
            print(f"Model: {[round(w,3) for w in model]}")
    return train_history, test_history, model_history


def predictionLabel(x, model, formal = False):
    """
    takes any features vector and model vector
    predicts the label based on the following logic -
        1. 1 if probability using sigmoid function > 0.5
        2. otherwise -1 (which represents 0)

    input vectors must be augmented to have the same dimension as model

    if formal = False, can only be applied to RDD where y values have been transformed to {-1, 1}

    for formal = True, applied to RDDs where y values are {0, 1}
    """

    prob = sigmoid(np.dot(model, x))
    if prob > 0.5:
        return [1, prob]
    else:
        if formal:
            return [0, prob]
        else:
            return [-1, prob]

#Transform y-values in the following way for compatibility with gradient descent function -
### y = 1 -> y' = 1
### y = 0 -> y' = -1

transformedTrain = trainRDD.map(lambda x: (2*x[0]-1, np.array(x[1]))).cache()
transformedTest = testRDD.map(lambda x: (2*x[0]-1, np.array(x[1]))).cache()

#apply augmentation
augTrain = transformedTrain.map(lambda x: (x[0], np.append([1.0], x[1]))).cache()
augTest = transformedTest.map(lambda x: (x[0], np.append([1.0], x[1]))).cache()

#Run gradient descent algorithm
wInit = np.array([0.25] + [0 for i in range(augTrain.take(1)[0][1].shape[0]-1)])
start = time.time()

#in GCP:
TrainLogLoss, TestLogLoss, Models = GradientDescent(augTrain, wInit, testRDD = augTest, nSteps = 550, learningRate = 0.0000001)

#in notebook
#TrainLogLoss, TestLogLoss, Models = GradientDescent(augTrain, wInit, testRDD = augTest, nSteps = 200, learningRate = 0.0001)

print(f"\n... trained {len(Models)} iterations in {time.time() - start} seconds")

#print loss values for plotting
print("\n LOSS DATA BELOW:")
print("Train Log Loss:")
print(TrainLogLoss)
print("\nTest Log Loss:")
print(TestLogLoss)


def modelMetrics(augTrain, augTest, model):
    '''prints all relevant metrics from model version
    input data must be augmented already'''

    #print outputs and some metrics
    print("\nFinal Model Vector:", model)

    # Predict Labels using both models
    GDPredsTrain = augTrain.map(lambda x: (x[0], predictionLabel(x[1], model))).cache()
    GDPredsTest = augTest.map(lambda x: (x[0], predictionLabel(x[1], model))).cache()


    #METRICS
    # Calculate Train Errors for comparison
    GDTrainError = GDPredsTrain.filter(lambda lp: lp[0] != lp[1][0]).count() / float(augTrain.count())

    TN = GDPredsTest.filter(lambda lp: lp[0] == lp[1][0]).filter(lambda lp: lp[0] == -1.0).count()
    TP = GDPredsTest.filter(lambda lp: lp[0] == lp[1][0]).filter(lambda lp: lp[0] == 1.0).count()
    FP = GDPredsTest.filter(lambda lp: lp[0] != lp[1][0]).filter(lambda lp: lp[0] == -1.0).count()
    FN = GDPredsTest.filter(lambda lp: lp[0] != lp[1][0]).filter(lambda lp: lp[0] == 1.0).count()

    Accuracy = float(TN+TP)/float(augTest.count())
    Precision = float(TP)/float(TP+FP)
    Recall = float(TP)/float(TP+FN)
    F1 =  2*(Recall*Precision)/(Recall+Precision)


    #Print Train Errors
    print("\nGradient Descent Training Error:",  str(GDTrainError))

    #Print Metrics on Test Data
    print("\n Prediction Metrics on Test Data")
    print("Accuracy:", Accuracy)
    print("Precision:", Precision)
    print("Recall:", Recall)
    print("F1:", F1)

    #Print Log Loss of Models
    print("\nGradient Descent Final Model Log Loss on Train Data:", LRLoss(augTrain, model))
    #Print Log Loss of Models
    print("\nGradient Descent Final Model Log Loss on Test Data:", LRLoss(augTest, model))


modelMetrics(augTrain, augTest, Models[-1])
