
# objective: extract customer perspectives on product from reviews

import json 
import pandas as pd 
import click
from top2vec import Top2Vec

from util.sentiments import sentiment_analysis
from util.topics import delight_attribute


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

    # pretrained embedding_model for generating word and document embeddings     
    model = Top2Vec(documents= reviews_df.body.values, speed='deep-learn', min_count=1, embedding_model='distiluse-base-multilingual-cased')

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
    