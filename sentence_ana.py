import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
import PyPDF2

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')

def analyze_sentiment(sentence):
    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)

    # Determine if the sentence is positive, negative, or neutral
    if sentiment_scores['compound'] >= 0.05:
        return "Positive", sentiment_scores
    elif sentiment_scores['compound'] <= -0.05:
        return "Negative", sentiment_scores
    else:
        return "Neutral", sentiment_scores

def analyze_file(file_path):
    # this function will analyze the sentence
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    sentences = sent_tokenize(content)

    print("________________________________________________________")
    print("File Content Analysis")
    overall_scores = {
        'positive': 0,
        'neutral': 0,
        'negative': 0,
        'compound': 0,
        'total_sentences': len(sentences)
    }

    for sentence in sentences:
        sentiment, s_score = analyze_sentiment(sentence)
        if sentiment == "Positive":
            overall_scores['positive'] += 1
        elif sentiment == "Negative":
            overall_scores['negative'] += 1
        else:
            overall_scores['neutral'] += 1
        overall_scores['compound'] += s_score['compound']

    print(f"Overall Sentiment: ")
    print(f"Positive: {overall_scores['positive']}")
    print(f"Negative: {overall_scores['negative']}")
    print(f"Neutral: {overall_scores['neutral']}")
    print(f"Average Compound Score: {overall_scores['compound'] / overall_scores['total_sentences']:.2f}")

    print("________________________________________________________")
    print("Sentence-by-Sentence Analysis")
    for sentence in sentences:
        sentiment, s_score = analyze_sentiment(sentence)
        print(f"Sentence: {sentence.strip()}")
        print(f"Sentiment: {sentiment}")
        print(s_score)
        print("--------------------------------------------------------")






if __name__ == "__main__":
    file_path = "abortion.txt"
    analyze_file(file_path)
