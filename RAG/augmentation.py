from retrieval import retrieval
from generation import chat

def prompt_creation_and_api_calls():
    query = "Compare the price of Samsung and Iphone Products please"
    #this gets the retrived products(combined and cleaned) from the query and creates the prompt
    augmented_document = retrieval(query)
    prompt = f"""
    These are the retrieved documents which represent a product description from various categories(seven) from daraz, an online shopping platform
    It represent the retrieved information of products from the given QUESTION from the qdrant database. based on the retrieved documents 
    provide me the information about the given QUESTION. However, you're talking to a customer who is asking about a specific topic.
    QUESTION: '{query}'
    PASSAGE: '{augmented_document}' 
    """

    #API CALL
    response = chat(prompt)
    print(response.text)