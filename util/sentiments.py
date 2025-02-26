import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

lang_processor = spacy.load("en_core_web_sm")

nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()


STARS_TO_SENTIMENT = {
    "1": "negative",
    "2": "negative",
    "3": "neutral",
    "4": "positive",
    "5": "positive",
}


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


if __name__ == "__main__":
    print(sentiment_analysis("5 stars"))