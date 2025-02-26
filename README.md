# delight_pointers

## Objective: 
from customer reviews on product, extract sentiment and product attribute that is being refferred to in the review for each review


## Setup

### Install Dependencies:

pip install -r requirements.txt

python -m spacy download en_core_web_sm

pip install top2vec[sentence_transformers]


### Run Command:

python review_analysis.py reviews.json --output output.csv