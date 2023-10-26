# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 12:35:53 2017

@author: rdonat

Program to traverse the directory structure of downloaded dataset and extract the data if interest: 
From, To, and the email body

"""

import os
from email.parser import Parser

rootdir = "C:\\Users\\rdonat\\Desktop\\MyFiles\\enron_mail_20150507\\maildir"

#Helper function used to perform email extraction
def email_extract(inputfile):
    #Open input file for reading    
    with open(inputfile, "r") as f:
        data = f.read()
    
    #Apply parser
    email = Parser().parsestr(data)
    
    #Delete extra characters from To list    
    if email['to']:
        email_to = email['to']
        email_to = email_to.replace("\n", "")
        email_to = email_to.replace("\t", "")
        email_to = email_to.replace(" ", "")
    
    #Delete extra characters from Email Body        
    email_text = email.get_payload()
    email_text = email_text.replace("\n", " ")
    #print((email['from']),'\n', (email_to),'\n', (email_text))
    
    #Open output file for writing
    with open("parsed_emails.txt", "a") as f:
        f.write(email['from'])
        f.write("*")
        try:        
            f.write(email_to)
        except:
            f.write("")
        f.write("*")
        f.write(email_text)
        f.write('\n')

#Loop through all subdirectories
for directory, subdirectory, filenames in  os.walk(rootdir):
   #print(directory, subdirectory, len(filenames))
    for filename in filenames:
        email_extract(os.path.join(directory, filename) )

 
