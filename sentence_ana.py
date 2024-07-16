import nltk # download sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Using sentiment analysis
nltk.download('vader_lexicon')

def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)

    # See if the sentence is possitive or negative
    if sentiment_scores['compound'] >= 0.05:
        return "Positive", sentiment_scores
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative",sentiment_scores
    else:
        return "Neutral",sentiment_scores

# Have input and output for python
sentence = input("Enter a sentence: ")
print("________________________________________________________")
print("Sentence Result")
sentiment, s_score = analyze_sentiment(sentence)
print(f"The sentiment of the sentence is: {sentiment}")
print(s_score)