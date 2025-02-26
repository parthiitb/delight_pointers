# objective: extract customer perspectives on product from reviews

import json 
import pandas as pd 
import click
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import spacy
import numpy as np 
from top2vec import Top2Vec


STARS_TO_SENTIMENT = {
    "1": "negative",
    "2": "negative",
    "3": "neutral",
    "4": "positive",
    "5": "positive",
}

lang_processor = spacy.load("en_core_web_sm")

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


def is_star_rated_review(review_text: str) -> tuple[bool, str]:
    """ analyze if review text directly contains star rating for the product, also returns star rating """
    
    text = review_text.lower().strip()

    # find if text contains a number and a word starting with st
    match_star = re.search(r"(\d)\s*(st[a-z]*)", text, re.IGNORECASE)
    
    if match_star:
        star_rating, star_word = match_star.groups()
        return True, star_rating
    return False, -1


def sentiment_analysis(review_text: str) -> str:
    """ Analyzes sentiment, considering both star ratings and text """
    
    is_star_rated, rating = is_star_rated_review(review_text)       # check if text directly contains stars rating
    
    if is_star_rated:
        # Separate the rating from the review text
        text_without_rating = re.sub(r"(\d)\s*st[a-z]*", "", review_text, flags=re.IGNORECASE).strip()        

        # when review contains only a rating (e.g., "5 stars"), use direct sentiment mapping
        if not text_without_rating:        
            sentiment = STARS_TO_SENTIMENT[rating]
            return sentiment
        
    score = sia.polarity_scores(review_text)
    sentiment = "positive" if score["compound"] > 0.05 else \
                "negative" if score["compound"] < -0.05 else \
                "neutral"
                
    return sentiment 


def delight_attribute(reviews_df, model):
    
    reviews_df['attribute'] = ""        
    topic_sizes, topic_nums = model.get_topic_sizes()
    topic_words, word_scores, topics = model.get_topics()
    
    for n in range(len(topic_nums)):    
        documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=n, num_docs=topic_sizes[n]) 
        reviews_df.loc[document_ids, ['attribute']] = topic_words[n][0]

    return reviews_df['attribute'].values


@click.command()
@click.argument("input_json", type=click.Path(exists=True))
@click.option("--output", default="output.csv", help="output file name")
def process_reviews(input_json, output):
    
    try:
        # reviews <- customer reviews of a product
        with open(input_json, "r") as data:
            reviews = json.load(data)                
    except Exception as e:
        print(f"Error while reading reviews json file: {e}")
     
                    
    # assign review sentiment               
    for review in reviews: 
        review['sentiment'] = sentiment_analysis(review['body'])

        
    reviews_df = pd.DataFrame(reviews)

    # using top2vec's pretrained embedding_model for generating joint word and document embeddings.     
    model = Top2Vec(documents= reviews_df.body.values, speed='deep-learn', min_count=1, embedding_model='distiluse-base-multilingual-cased')
    # model = Top2Vec(documents= reviews_df.body.values, speed='deep-learn', min_count=1, embedding_model='universal-sentence-encoder')

    # assign review delight attribute 
    reviews_df['attribute'] = delight_attribute(reviews_df, model)    

    
    # write delight attribute frequency for only positive reviews 
    positive_reviews = reviews_df[reviews_df['sentiment'] == "positive"]
    attribute_freq = positive_reviews['attribute'].value_counts()
    attribute_freq.to_csv(output)

    # print summary of successes v fails
    print("-----")
    print(f"Successes: {len(positive_reviews)}")
    print(f"Fails: {len(reviews_df) - len(positive_reviews)}")
    print("-----")  
    print("Done!")


if __name__ == "__main__":
    process_reviews()
    

    
# Notes:

    # Top2Vec automatically detects topics in text without requiring preprocessing like tokenization or stop-word removal.