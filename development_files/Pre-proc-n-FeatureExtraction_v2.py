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

# start Spark Session (RUN THIS CELL AS IS)
from pyspark.sql import SparkSession
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

rawTrainRDD = sc.textFile('gs://w261-tktruong/data/train.txt')
trainRDD, devRDD, testRDD = rawTrainRDD.randomSplit([0.8,0.1, 0.1], seed = 2018)
edaRDD, otherRDD = trainRDD.randomSplit([0.0003, 0.9997], seed = 2018)

#tenK_raw = ast.literal_eval(open("gs://w261-tktruong/eda.txt", "r").read())

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
    # "''"
    return row_values

# Calculate click through rate frequency count of each category

def BinCategoricalFeatures(tenK_df4):
    '''
    takes a spark df with numerical and categorical columns
    outputs a spark df where all the categorical features are binned using custom logic
    '''
    exclude_list = ['_20', '_31', '_37']

    tenK_click_df = tenK_df4
    for n,i in enumerate(tenK_df4.dtypes):

        if i[1]=='string':

            feature = i[0]

            # frequency count of unique categories under each feature
            cat_freqDF = tenK_df4.groupBy(feature).count()

            # click through frequency count: count of 'label = 1' for each category
            click_freqDF = tenK_df4.where("_1 == 1").groupBy(feature, "_1").count()


            ## Calculate click through frequency ratio for each category:
            ##(count of 'label = 1'/total count)

            df1 = click_freqDF.alias('df1')
            df2 = cat_freqDF.alias('df2')
            if n == 0:
                df3 = tenK_df4.alias('df3')
            else:
                df3 = tenK_click_df.alias('df3')

            tenK_click_df = df1.join(df2, [feature]).join(df3, [feature]).select(feature, 'df3.*',
                                    (df1['count']/df2['count']).alias(feature+"_click"))

            ## End of click through frequency ratio calculation

            ###### Bin data into binary bins based on the click through rate(ctr).

            if i[0] not in exclude_list:

                # if ctr == 0, value = A
                # else value = B
                # Keep null values as it is
                tenK_click_df = tenK_click_df.withColumn(feature,
                F.when(tenK_click_df[feature+'_click'] == 0, F.lit("A"))
                .otherwise(F.lit("B")))


            elif i[0] in ['_20', '_31']:

                max_ctr = tenK_click_df.agg({feature+"_click": "max"}).collect()[0][0]
                ctr_threshold = max_ctr/2

                # if ctr == 0, value = A
                # if ctr > 0 and <= threshhold, value = B
                # else value = C
                # Keep null values as it is
                tenK_click_df = tenK_click_df.withColumn(feature,
                F.when(tenK_click_df[feature+'_click'] == 0, F.lit("A"))
                .otherwise(
                    F.when((tenK_click_df[feature+'_click'] > ctr_threshold)|(tenK_click_df[feature+'_click'] > ctr_threshold)
                       , F.lit("B"))
                    .otherwise(F.lit("C"))))

            elif i[0] == '_37':

                max_ctr = tenK_click_df.agg({feature+"_click": "max"}).collect()[0][0]
                ctr_threshold1 = max_ctr/3
                ctr_threshold2 = 2*ctr_threshold1

                # if ctr == 0, value = A
                # if ctr > 0 and <= threshhold1, value = B
                # if ctr > threshhold1 and <= threshhold2, value = C
                # else value = D
                # Keep null values as it is

                tenK_click_df = tenK_click_df.withColumn(feature,
                F.when(tenK_click_df[feature+'_click'] == 0, F.lit("A"))
                .otherwise(
                    F.when(((tenK_click_df[feature+'_click'] > 0)
                            & ((tenK_click_df[feature+'_click'] < ctr_threshold1) | (tenK_click_df[feature+'_click'] == ctr_threshold1)))
                           , F.lit("B"))
                    .otherwise(
                        F.when(((tenK_click_df[feature+'_click'] > ctr_threshold1)
                            & ((tenK_click_df[feature+'_click'] < ctr_threshold2) | (tenK_click_df[feature+'_click'] == ctr_threshold2)))
                           , F.lit("C"))
                        .otherwise(F.lit("D")))))

    tenK_df5 = tenK_click_df.drop('_15_click','_16_click','_19_click','_22_click','_25_click','_27_click',
                                 '_28_click','_29_click', '_31_click', '_32_click', '_37_click', '_38_click'
                                 ,'_20_click','_23_click','_31_click', '_37_click')

    tenK_df5.cache()
    return tenK_df5

# FeatureScore calculation using RandomForest Ensembling

def CalFeatureScore(tenK_df5):
    '''
    Takes input as a Spark DataFrame.
    Fit and transfor using Assembler Pipeline
    Run RandomForestClassifier to output top performing 30 features
    '''

    def ExtractFeatureImp(featureImp, dataset, featuresCol):
        '''
        Function to display featureImportances in human readable format
        '''
        list_extract = []
        for i in dataset.schema[featuresCol].metadata["ml_attr"]["attrs"]:
            list_extract = list_extract + dataset.schema[featuresCol].metadata["ml_attr"]["attrs"][i]
        varlist = pd.DataFrame(list_extract)
        varlist['score'] = varlist['idx'].apply(lambda x: featureImp[x])
        return(varlist.sort_values('score', ascending = False))


    encoding_var = [i[0] for i in tenK_df5.dtypes if (i[1]=='string')]
    num_var = [i[0] for i in tenK_df5.dtypes if (i[1]!='string') & (i[0]!= '_1')]

    string_indexes = [StringIndexer(inputCol = c, outputCol = 'IDX_' + c, handleInvalid = 'keep')
                      for c in encoding_var]
    onehot_indexes = [OneHotEncoderEstimator(inputCols = ['IDX_' + c], outputCols = ['OHE_' + c])
                      for c in encoding_var]
    label_indexes = StringIndexer(inputCol = '_1', outputCol = 'label', handleInvalid = 'keep')
    assembler = VectorAssembler(inputCols = num_var + ['OHE_' + c for c in encoding_var]
                                , outputCol = "features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", seed = 8464,
                                 numTrees=10, cacheNodeIds = True, subsamplingRate = 0.7)

    pipe = Pipeline(stages = string_indexes + onehot_indexes + [assembler, label_indexes, rf])

    ## fit into pipe

    mod = pipe.fit(tenK_df5)
    tenK_df6 = mod.transform(tenK_df5)

    varlist = ExtractFeatureImp(mod.stages[-1].featureImportances, tenK_df6, "features")
    top_features = [x for x in varlist['name'][0:30]]

    return top_features

#Create data frame with one-hot encoding for categorical variables

def one_hot_encode(tenK_df5, top_features):
    '''
    Create data frame with one-hot encoding for categorical variables
    Take input as Spark Data Frame
    Output Spark DataFrame with hot-encoding
    '''

    one_hot = tenK_df5.toPandas()
    encoding_var = [i[0] for i in tenK_df5.dtypes if (i[1]=='string')]
    for col in encoding_var:
        one_hot_pd = pd.concat([one_hot,pd.get_dummies(one_hot[col], prefix='OHE_'+col,dummy_na=False)],axis=1).drop([col],axis=1)
        one_hot = one_hot_pd

    one_hot_df = sqlContext.createDataFrame(one_hot_pd)

    ###Keep the columns recommended by RandomForestClassifier

    curr_col = one_hot_df.columns
    col_to_drop = [x for x in curr_col if x not in top_features and x != '_1']

    tenK_df7 = one_hot_df
    for col in col_to_drop:
        tenK_df7 = tenK_df7.drop(col)

    return tenK_df7

# use average imputer for null values

def imputeNumeric(numeric_DF):
    '''
    takes a spark df with continuous numeric columns
    outputs a spark df where all null values are replaced with the column average

    the first column, which is the outcome values, are preserved
    '''
    outputColumns=["{}".format(c) for c in numeric_DF.columns[1:11]]
    catColumns = ["{}".format(c) for c in numeric_DF.columns[11:]]

    imputer = Imputer(
        inputCols=numeric_DF.columns[1:11],
        outputCols=["{}".format(c) for c in numeric_DF.columns[1:11]]
    )

    model = imputer.fit(numeric_DF)

    imputedDF = model.transform(numeric_DF).select(['_1']+outputColumns+catColumns)

    return imputedDF

def scaleFeatures(inputedDF):
    '''
    inputs imputed data frame with no null values and continuous features
    transforms the data frame into 2 column data frame with first column as label and second column as dense vector of features
    scales all features using the StandardScalar
    returns 2 column dataframe with scaled features
    '''

    transformedImputedDF = inputedDF.rdd.map(lambda x: (x[0], Vectors.dense(x[1:11]))).toDF(['label', 'x'])


    scaler = StandardScaler(inputCol="x",
                        outputCol="features",
                        withStd=True, withMean=True)

    scalerModel = scaler.fit(transformedImputedDF)
    scaledDF = scalerModel.transform(transformedImputedDF).select(['label', 'features'])

    return scaledDF


# parse raw 10k sample data to form tenKRDD
#tenKRDD = sc.textFile("gs://w261-tktruong/eda.txt").map(parse_raw_row).cache()

#### Create SQL dataframe from RDD
parsed_eda = edaRDD.map(parse_raw_row).cache()
# for 10K sample data
tenKfeature_df = sqlContext.createDataFrame(parsed_eda)

# drop features with high unknown values

tenK_df1 = tenKfeature_df.drop('_13','_36','_2','_11','_33','_34','_39','_40')

#tenK_df1.show(1)

tenK_df2 = tenK_df1.drop('_17','_18','_21','_24','_26','_30','_35')
#tenK_df2.show(5)

##Replace null with mean for numerical features

tenK_df4 = imputeNumeric(tenK_df2)
tenK_df4.cache()
#tenK_df4.show(1,False)

#### Customize binning for categorical features

tenK_df5 = BinCategoricalFeatures(tenK_df4)
#tenK_df5.show(20,False)

### Call RandomForest Classifier to retrieve top performing features
top_features = CalFeatureScore(tenK_df5)
print(top_features)

### Call one-hot encoding

tenK_df7 = one_hot_encode(tenK_df5, top_features)
#tenK_df7.show(5, False)

### Build separate RDD for Categorical columns

catDF = tenK_df7.select([c for c in tenK_df7.columns if 'OHE' in c ])
catRDD = catDF.rdd
#catRDD.take(5)

### Build separate RDD for Categorical columns

### Standardize numerical column and Build separate RDD for Numerical columns

numericDF = scaleFeatures(tenK_df7)
numRDD = numericDF.rdd
#numRDD.take(5)

### Combine both the RDD-s to build full data RDD
FullDataRDD = numRDD.zip(catRDD)

FullDataRDD1 =  FullDataRDD.map(lambda x: (x[0][0], np.array(x[0][1]), np.array(x[1])))\
                           .map(lambda x: (x[0], np.append(x[1], x[2])))

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
    
def predictionLabel(x, model):
    """
    takes any features vector and model vector
    predicts the label based on the following logic - 
        1. 1 if probability using sigmoid function > 0.5
        2. otherwise -1 (which represents 0)
        
    input vectors must be augmented to have the same dimension
    can only be applied to RDD where y values have been transformed to {-1, 1}
    """
    
    prob = sigmoid(np.dot(model, x))
    if prob > 0.5:
        return [1, prob]
    else:
        return [-1, prob]

# split 10K data set into train and test
# we use fullDataRdd1 above, which is the fully processed 10K set

#***************************************************************************************************#

TenKTrain, TenKTest = FullDataRDD1.randomSplit([0.8,0.2], seed = 2018)

#Transform TenKTrain, TenKTest for Gradient Descent Processing
#Transform y-values in the following way for compatibility with gradient descent function - 
### y = 1 -> y' = 1
### y = 0 -> y' = -1

transformed10KTrain = TenKTrain.map(lambda x: (2*x[0]-1, np.array(x[1]))).cache()
transformed10KTest = TenKTest.map(lambda x: (2*x[0]-1, np.array(x[1]))).cache()

#apply augmentation
aug10KTrain = transformed10KTrain.map(lambda x: (x[0], np.append([1.0], x[1]))).cache()
aug10KTest = transformed10KTest.map(lambda x: (x[0], np.append([1.0], x[1]))).cache()



#***************************************************************************************************#
#unregularized
wInit = np.array([0.25] + [0 for i in range(aug10KTrain.take(1)[0][1].shape[0]-1)])
start = time.time()
TrainLogLoss, TestLogLoss, TenKModels = GradientDescent(aug10KTrain, wInit, testRDD = aug10KTest, nSteps = 200, learningRate = 0.0001)
print(f"\n... trained {len(TenKModels)} iterations in {time.time() - start} seconds")


#***************************************************************************************************#
#validation using MLLib

TenKTrainDF = TenKTrain.map(lambda x: (x[0],Vectors.dense(x[1]))).toDF(['label', 'features'])

#fit logistic regression with no regularization
lr = LogisticRegression(maxIter=200, regParam=0, elasticNetParam=0)
lrModel = lr.fit(TenKTrainDF)

# Create model vectors for comparison
## gradient descent model from last iteration 
GDModelVector = TenKModels[-1]
## full MLLib Model with y-intercept as first term
MLLibModelVector = np.append(np.array(lrModel.intercept),np.array(lrModel.coefficients))

# Predict Labels using both models
GDPredsTrain = aug10KTrain.map(lambda x: (x[0], predictionLabel(x[1], GDModelVector)))
MLLibPredsTrain = aug10KTrain.map(lambda x: (x[0], predictionLabel(x[1], MLLibModelVector)))
GDPredsTest = aug10KTest.map(lambda x: (x[0], predictionLabel(x[1], GDModelVector)))
MLLibPredsTest = aug10KTest.map(lambda x: (x[0], predictionLabel(x[1], MLLibModelVector)))

# Calculate Train Errors for comparison
GDTrainError = GDPredsTrain.filter(lambda lp: lp[0] != lp[1][0]).count() / float(aug10KTrain.count())
MLLibTrainError = MLLibPredsTrain.filter(lambda lp: lp[0] != lp[1][0]).count() / float(aug10KTrain.count())

# Calculate Test Accuracies for comparison
GDTestAccuracy = GDPredsTest.filter(lambda lp: lp[0] == lp[1][0]).count() / float(aug10KTest.count())
MLLibTestAccuracy = MLLibPredsTest.filter(lambda lp: lp[0] == lp[1][0]).count() / float(aug10KTest.count())


#Print Train Errors
print("Gradient Descent Training Error = " + str(GDTrainError))
print("MLLib Training Error = " + str(MLLibTrainError))

#Print Test Accuracies
print("\nGradient Descent Test Accuracy = " + str(GDTestAccuracy))
print("MLLib Training Test Accuracy = " + str(MLLibTestAccuracy))

#Print Log Loss of Models
print("\nGradient Descent Final Model Log Loss on Train Data = ", LRLoss(aug10KTrain, GDModelVector))
print("MLLib Model Loss on Train Data = ", LRLoss(aug10KTrain, MLLibModelVector))

#Print Log Loss of Models
print("\nGradient Descent Final Model Log Loss on Test Data = ", LRLoss(aug10KTest, GDModelVector))
print("MLLib Model Loss on Test Data = ", LRLoss(aug10KTest, MLLibModelVector))

#Print Models for comparison:
print("\nGradient Descent Final Model= ", GDModelVector)
print("\nMLLib Model = ", MLLibModelVector)