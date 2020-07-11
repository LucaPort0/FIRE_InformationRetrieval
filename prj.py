import os
import string
import re
import time
import collections
import math

from nltk.corpus import stopwords
import pandas as pd 

# Printing function for nested dictionaries
def printTermsDict(d):
    for term, info in d.items():
        print(term, end = ': ')
        for fileName in info:
            print(fileName + ':', info[fileName])

# Function that prints a query dictionary
def printDict(d):
    for id, query in d.items():
        print(id,':', query)

# Dictionary sorting function 
def sortDict(d):
    od = collections.OrderedDict(sorted(d.items()))
    
    return od

#Normalizing and tokenizing function for the files
def normalizeFilesText(list, path):
    with open(path, "r") as f:
        list = [x.strip() for x in f.read().split()]
        
    stop = stopwords.words('english')

    for i in range (len(list)):
        if i != 1: #Skiping the first element(DocID) in the list
            list[i] = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ<>_]', '', list[i]) #Removing punctuation
        list[i] = re.sub("[<].*?[>]", "", list[i]) #Removing everything between <>
        list[i] = list[i].lower() #lowercasing elements
        if list[i] in stop:
            list[i] = ''
        
    while '' in list: #Remove null elements in the list 
        list.remove('')

    return list

# Function that read the query file and index it in a dictionary
def readQueries(path, d):
    with open(path, 'r') as f:
        text = [x.strip() for x in f.read().split()]

    text = ' '.join(text) #turning the text list into a string

    getID = "<num>(.*?)</num>" # Finding strings between <num>, in this case, the ids
    ids = re.findall(getID, text) 

    getQuery = "<narr>(.*?)</narr>"
    qrs = re.findall(getQuery, text)
    
    stop = stopwords.words('english')

    for i in range(len(ids)):
        qrs[i] = re.sub(u'[^a-zA-ZáéíóúÁÉÍÓÚâêîôÂÊÎÔãõÃÕçÇ<>_]', ' ', qrs[i]) # Normalizing
        qrs[i] = qrs[i].lower() # Lowercasing
        qrs[i] = list(qrs[i].split(' ')) # Turning the string back into a list
        for j in range(len(qrs[i])):
            if qrs[i][j] in stop:
                qrs[i][j] = ''
        
        while '' in qrs[i]: #Removing null elements in the list 
            qrs[i].remove('')

        d[ids[i]] = qrs[i] # Adding id and query to the dictionary
    return d


# Function that creates nested dictionaries of words
def createNestedDict(d, tokens):
    for i in range(1, len(tokens)):
        id = tokens[0] # ID gets the document id, always at position 0

        if tokens[i] not in d: # If the word is not in the dictionary
            d[tokens[i]] = {}  # A dictionary of that word is created, with the doc id and frequency 
            dict = tokens[i]
            dict = {
                id : 1
            } 
            d[tokens[i]] = dict
        else:
            if id in d[tokens[i]]: # If it is the same document 
                d[tokens[i]][id] += 1 
            elif id not in d[tokens[i]]: # If not, another document-frequency pair is created
                d[tokens[i]][id] = 1

    return d

# Function that opens all the utf8 files, normalizes the tokens and generate the dictionaries
def scan_folder(parent, tokens, d):
    # iterate over all the files in directory 'parent'
    for file_name in os.listdir(parent):
        if file_name.endswith(".utf8"):
            # if it's a utf8 files
            path = "".join((parent, "/", file_name)) # file name added to path
            global nDocs
            nDocs += 1
            createNestedDict(d, normalizeFilesText(tokens, path)) # nested dictionary is created or updated with tokens
            tokens = [] # token list is reseted
        else:
            current_path = "".join((parent, "/", file_name))
            if os.path.isdir(current_path):
                # if we're checking a sub-directory, recall this method
                scan_folder(current_path, tokens, d)
    return d

def tfIdf(term, termDict, nDocs, t):
    w = 0
    nt = len(termDict[term]) # Number of documents with "term"

    w = ((1 + math.log2(t))) * (math.log2(nDocs/nt))

    return w


# Vector Model function that returns the most relevant documents for the given query
def vectorModel(termDict, queryDict, queryID, nDocs, k):
    query = queryDict[queryID]
   # mostRelevant = []
    scores = {} # Dictionary that stores pairs of Document-Score
    length = {}
    wq = 0 # TfIdf of the term in the query
    wd = 0 # TfIdf of the term in the documents

    for qTerm in query: # Iterating through query
        tf = query.count(qTerm) # Counting the frequency of the term in the query

        wq = tfIdf(qTerm, termDict, nDocs, tf) 

        for doc, value in terms[qTerm].items():
            nt = value
            
            if doc not in scores:
                scores[doc] = 0
                length[doc] = 0

            wd = tfIdf(qTerm, termDict, nDocs, nt)
            scores[doc] += (wq * wd)
            length[doc] += (wd**2)

    for doc, value in scores.items():
        length[doc] = math.sqrt(length[doc])
        scores[doc] = (scores[doc]/length[doc])

    scores = sorted(scores.items(), key=lambda x: x[1], reverse = True)

    return scores[:k]

def evaluate(path):
    return 1

###########################################################################################

start_time = time.time()

tokens = [] # Token list
terms = {} # Dictionary of terms, behaving as an inverted list
queries = {} # Dictionary of queries and their ids
nDocs = 0 # Numero de documentos 
score = {} # Vetor de scores


#print('Insira o path do diretorio TELEGRAPH_UTF8')
#print('EX: "/home/user/Desktop/FIRE2010/en.doc.2010/TELEGRAPH_UTF8" ')
#files_path = input()

#print('Insira o path do arquivo com as queries')
#print('EX: "/home/user/Desktop/FIRE2010/en.topics.76-125.2010.txt" ')
#queries_path = input()

#print('Insira o número da comparação a ser feita:\n-1 Indexacao com e sem stopwords\n-2 Indexação com e sem radicalizacao\n-3 Comparacao entre modelo vetorial e probabilistico')
#operation = input()

print('Escolha o numero K de arquivos mais relevenates a serem recuperados')
k = int(input())

print('Escolha a query a ser analisada')
q = int(input())

terms = scan_folder("/home/user/Desktop/TopBD/Projeto/FIRE2010/en.doc.2010/TELEGRAPH_UTF8", tokens, terms)  # Insert parent directory's path

#terms = sortDict(terms) # Sorting the terms in the dictionary

queries = readQueries("/home/user/Desktop/TopBD/Projeto/FIRE2010/en.topics.76-125.2010.txt", queries)

score = vectorModel(terms, queries, '76', nDocs, k)

#evaluation = evaluate("/home/user/Desktop/TopBD/Projeto/FIRE2010/en.qrels.76-125.2010.txt")

print(' %s seconds ' % (time.time() - start_time))
