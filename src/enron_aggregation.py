#""" Classification Model for Enron Email Dataset  """

## Imports
from __future__ import print_function
import sys
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
import numpy as np
from operator import add

## CONSTANTS

APP_NAME = "My Spark Application"

    
## Main functionality

def main(sc, filename):
        
    #Read in the parsed_emails    
    rawdata = sc.textFile(filename, 20) 
    header = rawdata.first()
    rawdata = rawdata.filter(lambda x: x!= header).map(lambda x: x.split('*',2)).map(lambda w: (w[0], w[1]))
    rawdata.cache()
    print(rawdata.take(1))
    num_data = rawdata.count()
    print(num_data)

    
    #Read in spreadsheet data on insiders
    insiders = sc.textFile('/Users/rosadonat/temp/project/insider_financials.csv')
    header = insiders.first()
    insiders = insiders.filter(lambda x: x!=header)
    email_list = insiders.map(lambda s: s.split(',')).map(lambda x: (x[1])).collect()

    poi_list = insiders.filter(lambda a: a[16] == True).map(lambda s: s[1]).collect()
    print(poi_list)
    
    #Compute how any emails each user sent   
    FROMS = rawdata.map(lambda f: f[0]).map(lambda g: (g,1)).reduceByKey(add).filter(lambda h: h[0] in email_list)
    FROMS.cache()
    print(FROMS.take(3))
    print(FROMS.count())
    

    #Compute how many emails each user received
    TOS = rawdata.map(lambda r: r[1]).flatMap(lambda t: t.split(',')).map(lambda y: (y,1)).reduceByKey(add).filter(lambda u: u[0] in email_list)
    TOS.cache()
    print(TOS.take(3))
    print(TOS.count())
    
    
    #Compute how many emails each user sent to a POI
    
    #Helper function used to count the number of POI's in recipient list
    def count_tos_poi(key,value):
        count=0
        value_list = str.split(',')
        for i in value_list:
            if (i in poi_list):
                count +=1
        return(key, count)
    
    TO_POI = rawdata.map(lambda i: i).groupByKey().filter(lambda d: d[0] in email_list).map(lambda x: count_tos_poi(x[0],x[1]))
    TO_POI.cache()
    print(TO_POI.take(3))
    print(TO_POI.count())

    
    #Compute how many emails each user received from a POI
    FROM_POI = rawdata.map(lambda l: l).groupByKey().filter(lambda k: k[0] in poi_list) \
        .map(lambda x: x[1]).flatMap(lambda g: g.split(',')).map(lambda n: (n,1)).reduceByKey(add) \
        .filter(lambda f: f[0] in email_list)
    FROM_POI.cache()
    print(FROM_POI.take(3))
    print(FROM_POI.count())
    
    
    #Aggregate data back to insiders dataset and write RDD to ouput file
    aggregates12 = FROMS.join(TOS)
    aggregates34 = FROM_POI.join(TO_POI)
    aggregates = aggregates12.join(aggregates34)
    print(aggregates.first())

    insider_data = insiders.zip(aggregates).map(lambda x: (x[0][0:15], x[1][0:], x[16]))
    insider_data.saveAsTextFile('insider_data.csv')

    sc.stop()
  

##OTHER FUNCTIONS/CLASSES



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
