import re
import ast
import time
import itertools
import pyspark
import numpy as np
from numpy import allclose
import pandas as pd
#import matplotlib.pyplot as plt
from pyspark.sql import Row
from pyspark.sql import SQLContext
import pyspark.sql.functions as F
from pyspark.ml.feature import StandardScaler
from pyspark.sql.functions import *
from pyspark.ml.classification import  RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoderEstimator, VectorAssembler, VectorSlicer
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Imputer
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

# start Spark Session (RUN THIS CELL AS IS)
from pyspark.sql import SparkSession
sc = pyspark.SparkContext()
sqlContext = SQLContext(sc)

rawTrainRDD = sc.textFile('gs://w261-tktruong/data/train.txt')
trainRDD, devRDD, testRDD = rawTrainRDD.randomSplit([0.8,0.1, 0.1], seed = 2018)
edaRDD, otherRDD = trainRDD.randomSplit([0.0003, 0.9997], seed = 2018)

#tenK_raw = ast.literal_eval(open("gs://w261-tktruong/eda.txt", "r").read())


def feature_extraction(row, test_data = False, features=[]):
    '''
    Input: Take RDD as input and transform it into RDD to be used for model.
    test_data: a binary variable to indicate test or train data
    Output: Two RDD-s
    '''

    #ALL HELPER FUNCTIONS


    # Parse raw data
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


############################

    # parse raw train data to form trainFeatureRDD

    ######### To run in GCP #######
    trainFeatureRDD = row.map(parse_raw_row).cache()

    ######### To run in notebook
    #trainFeatureRDD = sc.parallelize(row).map(parse_raw_row).cache()
    #########

    #### Create SQL dataframe from RDD
    feature_df = sqlContext.createDataFrame(trainFeatureRDD)

    # drop features with high unknown values
    feature_df1 = feature_df.drop('_13','_36','_2','_11','_33','_34','_39','_40')

    ### Remove Categorical features with high % of Uniqueness of Categories
    feature_df2 = feature_df1.drop('_17','_18','_21','_24','_26','_30','_35')

    ##Replace null with mean for numerical features
    feature_df3 = imputeNumeric(feature_df2, 11)
    feature_df3.cache()

    #### Customize binning for categorical features
    feature_df4 = BinCategoricalFeatures(feature_df3)

    ### Use the top performing features recommended by RandomForest Classifier
    #top_features = CalFeatureScore(feature_df4)

    #### Call one-hot encoding
    feature_df5 = one_hot_encode(feature_df4, top_features)

    ####################################
    #### Format data to be used in model

    ### Build separate RDD for Categorical columns
    catDF = feature_df5.select([c for c in feature_df5.columns if 'OHE' in c ])
    catRDD = catDF.rdd

    ### Standardize numerical column and Build separate RDD for Numerical columns
    numericDF = scaleFeatures(feature_df5, 11)
    numRDD = numericDF.rdd

    ### Combine both the RDD-s to build full data RDD

    FullDataRDD = numRDD.zip(catRDD)
    FullDataRDD1 =  FullDataRDD.map(lambda x: (x[0][0], np.array(x[0][1]), np.array(x[1])))\
                               .map(lambda x: (x[0], np.append(x[1], x[2])))
    FullDataRDD2 = FullDataRDD1.map(lambda x: (x[0],Vectors.dense(x[1])))

    print(FullDataRDD1.take(5))
    print(FullDataRDD2.take(5))

    return(FullDataRDD1, FullDataRDD2)
##########################################################
############################################################
############# Start of code ############
    # Read the train and data file for feature extraction and data preparation

    ### In GCP ###
    rawTrainRDD = sc.textFile('gs://w261-tktruong/data/train.txt')
    rawTestRDD = sc.textFile('gs://w261-tktruong/data/test.txt')
    ###########
    ### In notebook ###
    #rawTrainRDD = ast.literal_eval(open("data/train.txt", "r").read())
    #rawTestRDD = ast.literal_eval(open("data/test.txt", "r").read())
    ############

    top_features = ['_7_imputed', '_14_imputed', '_6_imputed', '_8_imputed', '_9_imputed', '_10_imputed',
                    '_12_imputed', '_3_imputed', 'OHE__31_B', '_5_imputed', '_4_imputed', 'OHE__31_C',
                    'OHE__37_B', 'OHE__37_C', 'OHE__37_D', 'OHE__20_B', 'OHE__20_C', 'OHE__16_B',
                    'OHE__19_B', 'OHE__23_B', 'OHE__22_B', 'OHE__32_B', 'OHE__25_B', 'OHE__27_B',
                    'OHE__28_B', 'OHE__29_B', 'OHE__38_B', 'OHE__15_B']

    final_trainRDD1,final_trainRDD2 = feature_extraction(rawTrainRDD, features = top_features)
    final_testRDD1,final_testRDD2 = feature_extraction(rawTestRDD, features = top_features)

############# End of code ############
