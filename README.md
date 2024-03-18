# IR-model
You need a folder to save all of the file named in the doc_id_list  
Also for queries_id_list.txt
# step
* First :  
	* load your data into documentofall, a dictionary got key by each document id, and values by each document split()  
	* query as well  
* Second:   
	* calculate term frequency for both docs and queries, and get the inverse document frequency   
	* so we can calculate the TF-IDF of docs and queries  
* Third:  
	* get the cosine similarrity of every pairs of docs and queries as the score for judging  
* End:    
	* Each query will have 1000 documents sorted by score  
