""" Classification Model for Enron Email Dataset  """

## Imports
from __future__ import print_function
import sys
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import numpy as np
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.feature import StandardScaler
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.evaluation import BinaryClassificationMetrics

## CONSTANTS

APP_NAME = "My Spark Application"

    
## Main functionality

def main(sc, filename):
    # RECORD START TIME
    timestart = datetime.datetime.now()

    rawdata = spark.read.csv(filename, sep =',', encoding='UTF-8', comment=None, header=True, inferSchema=True)
    raw_data = rawdata.fillna(0)
    raw_data.show(1)
    
    #Select features
    rdd = raw_data.rdd.map(lambda x: (x[20], x[2], x[3], x[4], x[5], x[11], x[12], x[13], x[15], x[18], x[19]))
    print (rdd.first())
    rdd.cache()
    
   
    def parsePoint(line):
        values = [float(x) for x in line]
        return LabeledPoint(values[0], values[1:])
    
    parsedData = rdd.map(parsePoint)
    parsedData.cache()
    rdd.unpersist()
    
    ## Tuning model parameters

    # Creating training and testing sets to evaluate parameters
    train_data, test_data = parsedData.randomSplit([0.8, 0.2])
    train_data.cache()
    test_data.cache()
    tr = train_data.count()
    te = test_data.count()
    print(tr, te)
    parsedData.unpersist()
    
    # The parameter settings for the Decision Tree model:
    numClasses = [2] 
    categoricalFeaturesInfo = {}
    impurity = ['gini', 'entropy']
    maxDepth = [3, 5, 7]
    maxBins = [32]

    k = 5
    partitions = train_data.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2])
    
    #Start outer loop in which each parameter set is applied to the linear model (depends on number of parameters defined above) 
    for s in range(len(numClasses)):
        for r in range(len(impurity)):
            for m in range(len(maxDepth)):
                for n in range(len(maxBins)):
                    nC = numClasses[s]
                    imp = impurity [r]
                    mD = maxDepth[m]
                    mB = maxBins[n]
                  
                    #Start inner loop in which k-fold cross validation is applied (k times)
                    metrics = []    
                    for i in range(k):
                        for j in range(k):
                            if i != j:            
                                try:
                                    trainingSet = trainingSet.union(partitions[j])
                                except:
                                    trainingSet = partitions[j]
                            testSet = partitions[i]
                        print("New partitions created: ", trainingSet.count(), testSet.count())
                        metric, area = evaluate(trainingSet, testSet, nC, categoricalFeaturesInfo, imp, mD, mB)
                        metrics.append(metric)
                        del trainingSet
                        del testSet
                    print(metrics)
                    mean_error = sum(metrics)/k
                    try:
                        errors.append(mean_error)
                    except:
                        errors = []
                        errors.append(mean_error)
                    print('mean error = ', mean_error)
                    param_set = sc.parallelize([(mean_error, nC, imp, mD, mB)])
                    try:
                        param_grid = param_grid.union(param_set)
                    except:
                        param_grid = param_set
                    
                            
    print("Errors from k-fold cross validation (6 parameter sets using k=5): ", errors)                     
    min_error = min(errors)
    print("minimum error obtained: ", min_error)
    
    #Find the parameter set that produced the smallest error
    result = param_grid.filter(lambda keyvalue: keyvalue[0] == min_error).flatMap(lambda x: x).collect()
    print(result)
    nC = result[1]
    imp = result[2]
    mD = result[3]    
    mB = result[4]
    
    # PRINT ELAPSED TIME    
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds(), 2) 
    print ("Time taken to execute cross validation: " + str(timedelta) + " seconds")     
    
    # RECORD START TIME
    timestart2 = datetime.datetime.now()
    
    #Final run with optimal parameters
    final_metric, metrics = evaluate(train_data, test_data, nC, categoricalFeaturesInfo, imp, mD, mB)
        
    areaUnderPRCurve = metrics.areaUnderPR
    areaUnderROCCurve = metrics.areaUnderROC
    final_accuracy = (1-final_metric)*100
    
    print('Final Accuracy: ', final_accuracy, '%')
    # Area under precision-recall curve
    print("Area under PR = %s" % areaUnderPRCurve)
    # Area under ROC curve
    print("Area under ROC = %s" % areaUnderROCCurve)


    # PRINT ELAPSED TIME    
    timeend2 = datetime.datetime.now()
    timedelta2 = round((timeend2-timestart2).total_seconds(), 2) 
    print ("Time taken to execute final model train and test: " + str(timedelta2) + " seconds")     

    output = sc.parallelize([('DecisionTree', errors, result, final_accuracy, areaUnderPRCurve, areaUnderROCCurve, timedelta, timedelta2)])
    output.saveAsTextFile('/Users/rosadonat/temp/project/DecisionTree_output')    
    sc.stop()
  

##OTHER FUNCTIONS/CLASSES
  
def evaluate(train, test, numClasses, categoricalFeaturesInfo, impurity, maxDepth, maxBins):
    model = DecisionTree.trainClassifier(train, numClasses=numClasses, categoricalFeaturesInfo = {}, impurity=impurity, maxDepth=maxDepth, maxBins=maxBins)
    #tp = test.map(lambda p: (p.label, model.predict(p.features)))
    predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))        
    testErr = predictionAndLabels.filter(lambda v: v[0]!=v[1]).count() /float(test.count())
    metrics = BinaryClassificationMetrics(predictionAndLabels)
    return testErr, metrics
     
     
def abs_error(actual, pred):
    return np.abs(pred - actual)

def squared_error(actual, pred):
    return (pred - actual)**2

    

if __name__ == "__main__":
    # Configure OPTIONS
    conf = SparkConf().setAppName(APP_NAME)
    conf = conf.setMaster("local[*]")
    #in cluster this will be like
    #"spark://ec2-0-17-03-078.compute-#1.amazonaws.com:7077"
    sc   = SparkContext(conf=conf)
    spark = SparkSession(sc)
    filename = sys.argv[1]
    # Execute Main functionality
    main(sc, filename)



