""" Classification Model for Enron Email Dataset  """

## Imports
from __future__ import print_function
import sys
import datetime
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
#import numpy as np
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql import Row
import re


## CONSTANTS

APP_NAME = "My Spark Application"

    
## Main functionality

def main(sc, filename):
        
    ################################################################################################
    #
    #   Import Rawdata
    #
    ################################################################################################
    
    # RECORD START TIME
    timestart = datetime.datetime.now()
    
    rawdata = sc.textFile(filename, 20) 
    emails = rawdata.map(lambda x: x.split('*', 2)).map(lambda d: d[2])
    print(emails.first())
    header = ['email']

    #Helper function to convert RDD to Rows
    def list_to_row(key, value):
        row_dict= dict(zip(key,value))
        return Row(**row_dict)
    
    rdd_rows = emails.map(lambda x: list_to_row(header, [x]))
    print(rdd_rows.take(3))
    
	#Convert Rows to Dataframe
    df = spark.createDataFrame(rdd_rows)
    df.show(3)
	#Fill in missing elements with empty string
    df = df.fillna({'email': ''}) 

	#Helper function to remove stopwords, punctuation, and change to lower case
        
    def clean_text(record):
        text  = record[0]
        words = text.split()
        
        # Default list of Stopwords
        stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', u'can', 'cant', 'come', u'could', 'couldnt', u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', u'each', u'few', 'finally', u'for', u'from', u'further', u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', u'just', u'll', u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', u'no', u'nor', u'not', u'now', u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over',u'own', u'r', u're', u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', u'under', u'until', u'up', u'very', u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', u'y', u'you', u'your', u'yours', u'yourself' u'yourselves']
        
        # Option to add more stopwords 
        stopwords_custom = ['']
        stopwords = stopwords_core + stopwords_custom
        stopwords = [word.lower() for word in stopwords]    
        
        # Remove special characters
        output = [re.sub('[^a-zA-Z0-9]','',word) for word in words]  
        # Change to lower case, remove stopwords and words less than 2 characters                                  
        output = [word.lower() for word in output if len(word)>2 and word.lower() not in stopwords]     
        return output
    
    udf_cleantext = udf(clean_text , ArrayType(StringType()))
    D1 = df.withColumn("words", udf_cleantext(struct([df[x] for x in df.columns])))
    D1.show(3)
    
	
    #Generate TF Count Vectorizer to obtain the Term Document Matrix
    CV = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 1000)
    CVmodel = CV.fit(D1)
    vectorData = CVmodel.transform(D1)
    
    vocab = CVmodel.vocabulary
    vocab_broadcast = sc.broadcast(vocab)
    
    IDF_ = IDF(inputCol="rawFeatures", outputCol="features")
    IDFModel = IDF_.fit(vectorData)
    scaledData = IDFModel.transform(vectorData)
    vectorData.show(10)
    scaledData.show(10)
    
    #Apply LDA clustering
        
    LDA_ = LDA(k=25, seed=123, optimizer="em", featuresCol="features")
    
    LDAmodel = LDA_.fit(scaledData)
    
    #Show topics, although they are still numeric values
    lda_topics = LDAmodel.describeTopics()
    lda_topics.show(25)
    
	#Helper function to map the term ID's back to words
    def mapper(termIndices):
        words = []
        for termID in termIndices:
            words.append(vocab_broadcast.value[termID])
        return words
    
	#Show the topics as words 
    udf_map_termID_to_Word = udf(mapper, ArrayType(StringType()))
    lda_topics_mapped = lda_topics.withColumn("topic_desc", udf_map_termID_to_Word(lda_topics.termIndices))
    lda_topics_mapped.select(lda_topics_mapped.topic, lda_topics_mapped.topic_desc).show(25,False)
    
    # PRINT ELAPSED TIME
    timeend=datetime.datetime.now()
    timedelta=(timeend-timestart).total_seconds()
    print('Time take to extract topics: ' + str(timedelta) + ' seconds')
    
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


