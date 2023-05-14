import datetime
import os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

lexicom = set()   #all different word in doc & query


with open("docs_id_list.txt") as f:
    all_doc = f.read.splitlines()

documentofall = {}
for filename in all_doc:
    with open('documents/'+ filename + '.txt', 'r',encoding="utf-8") as f:
        text = f.read()
        documentofall[filename] = text.split()
        #lexicom = lexicom.union(set(documentofall[filename])) #in order to make the system more efficient

with open('queries_id_list') as f:
    all_query = f.read.splitlines()

queryofall = {}
for filename in all_query:
    with open('queries/' + filename +'.txt', 'r',encoding="utf-8") as f:
        text = f.read()
        queryofall[filename] = text.split()
        lexicom = lexicom.union(set(queryofall[filename]))

lexicom = list(lexicom)

#term frequency of document
tf_list_of_doc = []
for item in documentofall.values():
    tf_of_each_doc = []
    for voc in lexicom:
        tf_of_each_doc.append(item.count(voc))
    tf_list_of_doc.append(tf_of_each_doc)

#term frequency of query
tf_list_of_query = []
for item in queryofall.values():
    tf_of_each_query = []
    for voc in lexicom:
        tf_of_each_query.append(item.count(voc))
    tf_list_of_query.append(tf_of_each_query)

#document frequency
df = []
for word in lexicom:
    count = 0
    for item in documentofall.values():
        if word in item:
            count +=1
    df.append(count)

# inverse document frequency
idf = []
doc_length = len(documentofall)
for fre in df:
    idf.append(np.log(doc_length/fre))

#tf-idf(doc)
tf_idf_doc = []
temp = []
for termfreq in tf_list_of_doc:
    temp = [x*y for x,y in zip(termfreq,idf)]
    tf_idf_doc.append(temp) 
print(len(tf_idf_doc))
#tf-idf(query)
tf_idf_query = []
temp = []
for termfreq in tf_list_of_query:
    temp = [x*y for x,y in zip(termfreq,idf)]
    tf_idf_query.append(temp) 
print(len(tf_idf_query))

#cosine similarity
cosine_of_doc_query = []
cosine_of_doc_query = cosine_similarity(tf_idf_doc, tf_idf_query)

#doc_list
with open('docs_id_list.txt') as f:
    docs_id_list = f.read().splitlines()

#query_list
with open('queries_id_list.txt') as f:
    query_id_list = f.read().splitlines()

#panda Dataframe for similarity
twodarray = pd.DataFrame(cosine_of_doc_query)
twodarray.index = docs_id_list
twodarray.columns = query_id_list
print(twodarray)
#result
file = 'result' + '_' + now.strftime("%y%m%d_%H%M") +'.csv'

with open(file, "w",encoding="utf-8") as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_id_list:
        f.write(query + ",")
        listedscore = twodarray[query].sort_values(ascending=False)
        f.write(' '.join(listedscore.index.to_list())+"\n")
    
