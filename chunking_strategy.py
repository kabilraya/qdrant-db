import os
import json
import re

#this function just loads the json file and filter out the 
# listItems from the dictionary
def get_filtered_data(file_name):
    with open(file_name,'r',encoding="utf-8") as f:
        data = json.load(f)
    #get only the ListItems
    cleaned_data = data["mods"]["listItems"]
    return cleaned_data

#this function takes in one item from the item list and chunks them
def chunk_each_docs(doc,chunk_size = 128):
    start = 0
    doc = json.dumps(doc)
    tokens = re.findall(r'\w+|[{}[\]:,",]',doc)
    chunks = []
    while start<=len(tokens):
        end = min(start+chunk_size,len(tokens))
        chunk = " ".join(tokens[start:end])
        chunks.append(chunk)
        start+=chunk_size
    return chunks

#this takes the dictionary(only listItems) and 
# iterates over it one item at a time
def get_chunks_of_items(filename):
    items_list = get_filtered_data(filename)
    all_chunks = []
    for item in items_list:
        chunks_of_each_docs = chunk_each_docs(item)
        all_chunks.append(chunks_of_each_docs)


