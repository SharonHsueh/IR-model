import os
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

lexicom = set()   #all different word in doc & query

doc_path = r"C:\Users\user\dataHW2\documents"
all_doc = os.listdir(doc_path)
documentofall = {}
for filename in all_doc:
    with open(os.path.join("documents", filename), 'r',encoding="utf-8") as f:
        text = f.read()
        documentofall[filename] = text.split()
        #lexicom = lexicom.union(set(documentofall[filename])) #in order to make the system more efficient

#average document length
count = 0
for item in documentofall.values():
    count+=len(item)
avg_doc_len = count/5000

query_path = r"C:\Users\user\dataHW2\queries"
all_query = os.listdir(query_path)
queryofall = {}

for filename in all_query:
    with open(os.path.join("queries", filename), 'r',encoding="utf-8") as f:
        text = f.read()
        queryofall[filename] = text.split()
        lexicom = lexicom.union(set(queryofall[filename]))

lexicom = list(lexicom)

#term frequency of document
k1 = 2
b = 0.5
tf_list_of_doc = []
for item in documentofall.values():
    tf_of_each_doc = []
    for voc in lexicom:
        tf_of_each_doc.append(item.count(voc))
    tf_of_each_doc = [3*i/((1)+1*len(item)/avg_doc_len+i)for i in tf_of_each_doc]#Document term weighting
    tf_list_of_doc.append(tf_of_each_doc)

#term frequency of query
tf_list_of_query = []
for item in queryofall.values():
    tf_of_each_query = []
    for voc in lexicom:
        tf_of_each_query.append(item.count(voc))
    tf_list_of_query.append(tf_of_each_query)

#Document don't contain term w
num_of_w = []
for word in lexicom:
    count = 0
    for item in documentofall.values():
        if word in item:
            count +=1
    num_of_w.append(count)
new_num_of_w = [(5000-i+0.5)/(i+0.5)for i in num_of_w]

#BM25
    
#query term weighting
k3 = 1000
new_tf_list_of_query = []
for item in tf_list_of_query:
    new_tf_list_of_query_temp = [i*(k3+1)/(i+k3) for i in item]
    new_tf_list_of_query_temp2 = [x*y for x,y in zip(new_tf_list_of_query_temp,new_num_of_w)]
    new_tf_list_of_query.append(new_tf_list_of_query_temp2)

BM25_similarity = []
for item in new_tf_list_of_query:
    BM25temp = []
    for item2 in tf_list_of_doc:
        temp = [x*y for x,y in zip(item, item2)]
        total = sum(temp)
        BM25temp.append(total)
    BM25_similarity.append(BM25temp)

#doc_list
with open("docs_id_list.txt") as f:
    docs_id_list = f.read().splitlines()
#query_list
with open("queries_id_list.txt") as f:
    query_id_list = f.read().splitlines()

#panda Dataframe for similarity
twodarray = pd.DataFrame(BM25_similarity)
twodarray = twodarray.transpose()
twodarray.index = docs_id_list
twodarray.columns = query_id_list
print(twodarray)
#result
now = datetime.datetime.now()
file = 'results/result' + '_' + now.strftime("%y%m%d_%H%M") + '.txt'
print(file)
with open(file, "w",encoding="utf-8") as f:
    f.write('Query,RetrievedDocuments\n')
    for query in query_id_list:
        f.write(query + ",")
        listedscore = twodarray[query].sort_values(ascending=False)
        f.write(' '.join(listedscore.index.to_list())+"\n")
