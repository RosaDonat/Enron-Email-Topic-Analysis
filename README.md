# Problem Statement:

Use Spark Machine Learning tools to analyze data in the Enron Email Dataset and identify those who may be Persons of Interest in the fraud investigation.

# Dataset:

I obtained the dataset from the following URL: https://www.cs.cmu.edu/~enron/. The data was collected during the Enron fraud investigation that took place in 2001 and has been used for educational purposes since then. It contains emails from about 150 users, mostly senior management of Enron, organized into folders. There are about 500 thousand messages in total, and they are not very lengthy documents, with a total dataset size of about 1.2GB.

I relied on external news sources that could provide me with the data wasn’t available when the investigation was in progress which is: who are the Enron employees that were charged with crimes and which of them were convicted of those crimes. This  USA Today Article details the fraud charges and convictions, and this data was useful to me.

I also obtained an Enron Statement of Financial Affairs spreadsheet from Find Law, a website that distributes legal information. This document contains financial data such as salary, bonus, and total stocks, for over 140 Enron employees.
Once I obtained these two additional sources of information, I could see that there was some overlap in the names from the article, the names in the financial data, and the names in the email folders.  

This allowed me to focus my search on those individuals that I had financial data for and those that are already identified as POI’s.  With such a massive number of emails, it was important to narrow the search.  I gathered the spreadsheet data into a .csv file and included the POI (person of interest) feature as well as the email address of each person, the file is called insider_financials.csv.

## Step 1: Email Pre-Processing
Each user in the email dataset had their own folder, with a list of sub-folders as one would normally see in an email program; most users had only the default folders, but some had quite a number of custom folders. 

In order to traverse these thousands of folders, I used a Python utility called os.walk which was incredibly useful. To extract the email data I used a Python utility called email.parser that was essentially designed for this purpose. Using these together the program starts at the root directory and traverses through all of the subdirectories and extracts the raw text from the emails contained. The email parser function tags all of the data in the header, and treats the email body as the ‘payload’. 

I used this function to extract the fields of interest for the aggregation task which are From, To, and Email Body. There is only one sender of each email, but there can be many recipients. These are captured in one field, which I will split in later steps for counting. 

## Step 2: Topical analysis of the emails – finding trends in the data
The last step in the Enron dataset analysis is the topical analysis of the emails themselves. There are over 500,000 emails in the corpus, and they have already been parsed out in the email pre-processing step, so they are in a .txt file already.
The body of the email is the third field in the parsed email dataset, each email is a document, regardless of how short it is.

In order to extract the features, I created a Term Document Matrix in which each row corresponds to a document and each column is a unique word in the document collection. The entries in this matrix are tf-idf “scores” that are calculated for each word. The tfidf score is the product of these two terms:
#### tf = term frequency = frequency of the term/total # of terms in the document
#### idf = inverse document frequency = log(total # of documents/number of documents with term in it)
This approach assigns less weight to very common terms and more weight to unique terms that appear rarely. This way the algorithm can more readily identify the important words in the document.

In my code I used CountVectorizer to learn the vocabulary and the term frequency, then I use the IDF library to obtain the inverse document frequency and generate the Term Document Matrix, in which scores are assigned to the document terms according to the learned vocabulary.

Once the Document Term Matrix is generated I applied clustering using Latent Dirichlet allocation (LDA). LDA is a topic model which infers topics from a collection of text documents. It can be thought of as a clustering algorithm as follows:
-	Documents correspond to records, topics correspond to cluster centers.
-	Topics and documents both exist in a feature space, where the feature vectors are vectors of word counts.
-	Rather than estimating a clustering using a traditional distance, LDA uses a function based on a statistical model of how text documents are generated.
The selection of k, the number of topics, was a challenge because there was no way to know how many topics there may be in the dataset. I chose a large number, k=50 even though the true number of existing topics may be higher. 
