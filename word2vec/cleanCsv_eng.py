#!/usr/bin/env python
# coding: utf-8


import csv
import re

csv.field_size_limit(100000000)

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`*$@#]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
   # string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
   # string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
   # string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = string.lower()
    #reducing multiple consecutive occurrences to two occurrences
    #string = re.sub(r'([A-Za-z])\1{2,}',r'\1\1', string)
    string = re.sub(r'(.)\1{2,}',r'\1\1', string)
    return string.strip().lower()


CSV_FILE_NAME = ""


def createNewCsv(file_path = CSV_FILE_NAME):
    with open(file_path, 'w+') as writeFile:
        pass

def appendListToCsv(list_csv, file_path):
    with open(file_path, 'a', encoding='utf-8') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(list_csv)

def getListFromCsvV2(filePath):
    with open(filePath, 'r', encoding='utf-8') as readFile:
        lines = [line.rstrip('\n') for line in readFile]
        return lines

# In[46]:


import pandas as pd

chunksize = 30000
#csvFilePath = "/data/ourdata/backup/commentonly_2000.csv"
#csvFilePath = "/data/ourdata/backup/commentonly.csv"
csvFilePath = "rt-polarity.pos_9Jan"

#csvOutputFilePath = "commentsonly_2000_eng_consec_3Jan_cleaned.csv"
#csvOutputFilePath = "commentsonly_eng_consec_3Jan_cleaned.csv"
csvOutputFilePath = "rt-polarity.pos_9Jan_cleaned"

createNewCsv(csvOutputFilePath)

i=0
for chunk in pd.read_csv(csvFilePath, chunksize=chunksize):
#for chunk in pd.read_csv(csvFilePath, chunksize=chunksize, error_bad_lines=False):
    i+=1
    print ('iteration: ', i)

    list_chunk = chunk.iloc[:,0].tolist()
    
    csv_output_list = [[clean_str(str(sent))] for sent in list_chunk]
    print('length csv_output_list : ', len(csv_output_list))
    
    csv_output_list = [x for x in csv_output_list if x[0]]
    print('length csv_output_list rem blank : ', len(csv_output_list))
    
    appendListToCsv(csv_output_list, csvOutputFilePath)
    
