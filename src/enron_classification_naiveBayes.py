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
from pyspark.mllib.classification import NaiveBayes, NaiveBayesModel
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
    print (rdd.take(10))
    rdd.cache()
    
    needs_scaling = rdd.map(lambda x: (x[1:]))
    scaler = StandardScaler(withMean = False, withStd = True).fit(needs_scaling)
    # Create new RDD called scaledData that contains scaled values of numerical values
    scaledData = rdd.zip(scaler.transform(needs_scaling)).map(lambda x: (x[0][0],x[1][0],x[1][1],x[1][2],x[1][3],x[1][4],x[1][5],x[1][6],x[1][7],x[1][8], x[1][9]))    
    scaledData.cache()   
    print(scaledData.take(10))
   
    def parsePoint(line):
        values = [abs(float(x)) for x in line]
        return LabeledPoint(values[0], values[1:])
    
    parsedData = scaledData.map(parsePoint)
    parsedData.cache()
    scaledData.unpersist()
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
    
    # The parameter settings for the Naive Bayes classification model:
    rate = [0.01, 0.1, 1.0]
    k = 5
    partitions = train_data.randomSplit([0.2, 0.2, 0.2, 0.2, 0.2])
    
    #Start outer loop in which each parameter set is applied to the linear model (depends on number of parameters defined above) 
    for s in range(len(rate)):
        learning_rate = rate[s]
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
            metric, area = evaluate(trainingSet, testSet, learning_rate)
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
        param_set = sc.parallelize([(mean_error, learning_rate)])
        try:
            param_grid = param_grid.union(param_set)
        except:
            param_grid = param_set
                            
                            
    print("Errors from k-fold cross validation (2 parameter sets using k=5): ", errors)                     
    min_error = min(errors)
    print("minimum error obtained: ", min_error)
    
    #Find the parameter set that produced smallest error
    result = param_grid.filter(lambda keyvalue: keyvalue[0] == min_error).flatMap(lambda x: x).collect()
    print(result)
    learning_rate = result[1]
        
    # PRINT ELAPSED TIME    
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds(), 2) 
    print ("Time taken to execute cross validation: " + str(timedelta) + " seconds")     
    
    # RECORD START TIME
    timestart2 = datetime.datetime.now()
    
    #Final run with optimal parameters
    final_metric, metrics = evaluate(train_data, test_data,learning_rate)
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

    output = sc.parallelize([('NaiveBayes', errors, result, final_accuracy, areaUnderPRCurve, areaUnderROCCurve, timedelta, timedelta2)])
    output.saveAsTextFile('/Users/rosadonat/temp/project/NaiveBayes_output')    
    sc.stop()
  

##OTHER FUNCTIONS/CLASSES
  
def evaluate(train, test, learning_rate):
	model = NaiveBayes.train(train, lambda_=learning_rate)
	#tp = test.map(lambda p: (p.label, model.predict(p.features)))
	predictionAndLabels = test.map(lambda lp: (float(model.predict(lp.features)), lp.label))        
	testErr = predictionAndLabels.filter(lambda s: s[0]!=s[1]).count() /float(test.count())
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



