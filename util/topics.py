
import numpy as np 


# Notes:
# ---
# using Top2Vec to automatically detect topics in text 
# without requiring preprocessing like tokenization or stop-word removal.

def delight_attribute(reviews_df, model):
    
    reviews_df['attribute'] = ""        
    topic_sizes, topic_nums = model.get_topic_sizes()
    topic_words, word_scores, topics = model.get_topics()
    
    for n in range(len(topic_nums)):    
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=n, num_docs=topic_sizes[n]) 
        reviews_df.loc[document_ids, ['attribute']] = topic_words[n][0]

    return reviews_df['attribute'].values
