from flask import Flask, jsonify, Response,request, json
from flask import request
import pandas as pd
import logging
import chardet
import re
import nltk
import csv
import time
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
stop_words = set(stopwords.words('indonesian'))
from nltk.sentiment.util import mark_negation   
from nltk.tokenize import word_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from nltk.tokenize import RegexpTokenizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
from nltk.tokenize import RegexpTokenizer
import numpy as np
import urllib3
from flask_cors import CORS
from sklearn.preprocessing import MinMaxScaler
import joblib
from scipy.sparse import hstack
import math
app = Flask(__name__)
app.debug = True
# Configure logging
logging.basicConfig(filename='error.log', level=logging.DEBUG)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

#API_ENDPOINT = 'http://127.0.0.1:5000/preprocessing'

model_dataset = joblib.load('./model/model_dataset.pkl')
vectorizer_dataset = joblib.load('./vectorizer/vectorizer_dataset.pkl')
model_AI = joblib.load('./model/model_datasetAI.pkl')
vectorizerAI = joblib.load('./vectorizer/vectorizer_datasetAI.pkl')


app = Flask(__name__)
CORS(app)
##REMOVE EMOJI FUNCTION
emoji_pattern = re.compile("["
                u"\U0001F000-\U0001F9EF"  # Miscellaneous Symbols and Pictographs
                u"\U0001F300-\U0001F5FF"  # Symbols and Pictographs Extended-A
                u"\U0001F600-\U0001F64F"  # Emoticons (Emoji)
                u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                u"\U0001F700-\U0001F77F"  # Alchemical Symbols
                u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-B
                u"\U00002702-\U000027B0"  # Dingbats
                u"\U0001F018-\U0001F270"  # Miscellaneous Symbols
                u"\u200d"  # Zero Width Joiner
                u"\u200c"  # Zero Width Non-Joiner
                "\U0001F602"
                "]+", flags=re.UNICODE)
def remove_emojis(text):
    if isinstance(text, str):
        return emoji_pattern.sub(r'', text)
    else:
        return text
################################################################
#CLEANSING AND REMOVE PUNCTUATION AND CASEFOLDING
def cleansing(text):
    if isinstance(text, float):
        text = str(text)
    text = text.strip()
    text = re.sub(r"[?|$|.|~!_:')(-+,]", '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\b[a-zA-Z]\b', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub('RT', '', text)
    text = re.sub('@[^\s]+', '', text)
    text = re.sub(r'https\S+', '', text)
    text = re.sub(r'#\w+\s*', '', text)
    #text = remove_emojis(text)
    text = re.sub(r'\u2026', '', text)
    text = re.sub('/', '', text)
    text = re.sub('%', '', text)
    text = re.sub('-', '', text)
    text = re.sub("'", '', text)
    text = re.sub("#", '', text)
    text = re.sub("—", '', text)
    text = re.sub("ðŸ¤”", '', text)
    text = re.sub("ðŸ¤¦", '', text)
    text = re.sub("£", '', text)
    text = re.sub("â€“", '', text)
    text = re.sub("ðŸ¤£", '', text)
    text = re.sub("ðŸ¤—", '', text)
    text = re.sub("ðŸ¥°", '', text)
    text = re.sub("@", '', text)
    text = re.sub("ðŸ¤§", '', text)
    text = re.sub("ðŸ¥°", '', text)
    text = re.sub("ðŸ§‘â€ðŸ¦¯", '', text)
    text = re.sub("ðŸ¤²", '', text)
    text = re.sub("ðŸ¤®", '', text)
    text = re.sub("donkâ€¼ï", '', text)
    text = re.sub("ðŸ¤®", '', text)
    return text
########################################################################
def case_folding(text):
    if isinstance(text, str):
        text = text.lower()   
    return text
################################################################
# Tokenize text
def tokenize(text):
    return nltk.word_tokenize(text)
################################################################
# Remove stopwords
def remove_stopwords(tokens):
    filtered_tokens = [word for word in tokens if word not in stopwords.words('indonesian')]
    return filtered_tokens
################################################################
# Replace slang words with real words
slang_dict = pd.read_csv("slang/slanglist.csv", index_col=0)["real_word"].to_dict()
def replace_slang(tokens):
    slang_words = [word for word in tokens if word in slang_dict]
    replaced_words = [slang_dict[word] if isinstance(slang_dict.get(word), str) else '' for word in slang_words]
    tokens = [replaced_words[slang_words.index(word)] if word in slang_words else word for word in tokens]
    return tokens
# Preprocess the text data
def preprocess_text(text):
    # Tokenize the text
    tokens = tokenize(text)
    # Apply mark negation
    tokens = mark_negation(tokens)
    # Replace slang words
    tokens = replace_slang(tokens)
    # Join the tokens back into a string
    text = " ".join(tokens)
    return text
################################################################
# Stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# Tokenize and stem text
def tokenize_and_stem(text):
    tokens = word_tokenize(text)  # Tokenize using nltk.word_tokenize
    stems = [stemmer.stem(token) for token in tokens]
    return stems

# Load the slang dictionary
slang_dict = pd.read_csv("slang/slanglist.csv", index_col=0)["real_word"].to_dict()

# Define a function to replace slang words with their real counterparts
def replace_slang(text):
    pattern = re.compile(r'\b(' + '|'.join(slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

# Define a function to preprocess the text data
def slang_remove(text):
    # Replace slang words with their real counterparts
    text = replace_slang(text)
    # Tokenize the text
    text = nltk.word_tokenize(text)
    # Apply mark negation
    text = mark_negation(text)
    # Join the tokens back into a string
    text = " ".join(text)
    return text


#################
# Create a regex tokenizer
tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')

# Initialize the CountVectorizer
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words, ngram_range=(1, 1))

def vectorize_text(text):
    # Apply CountVectorizer to the preprocessed data
    text_features = vectorizer.transform(text)
    return text_features

# Initialize label encoder
label_encoder = LabelEncoder()
positive_emotions = ['senang', 'terkejut', 'netral']
negative_emotions = ['sedih', 'takut']

@app.route('/preprocessing', methods=['POST'])
def pre_processing():
    json_data = request.get_json()
    data = json_data.get('data')
    
    remove_emoji = []
    remove_character = []
    text_lower_case = []
    tokens = []
    stopwords_list = []
    stemmed_tokens = []
    stemmed_text = []
    slang_removed_text = []
    final_process = []
    
    
    for tweet in data:
        modified_tweet = {
            'casefolding_text': case_folding(tweet.get('full_text')),
            'original_casefolding' :tweet.get('full_text')
        }   
        text_lower_case.append(modified_tweet)
    
    for tweet in text_lower_case:
        modified_tweet = {
            'removeemoji_text': remove_emojis(tweet.get('casefolding_text')),
            'original_emoji' : tweet.get('casefolding_text')
        }
        remove_emoji.append(modified_tweet)
    
    for tweet in remove_emoji:
        modified_tweet = {
            'cleansing_text': cleansing(tweet.get('removeemoji_text')),
            'original_cleansing' : tweet.get('removeemoji_text')
        }
        remove_character.append(modified_tweet)
        
    for tweet in remove_character:
        modified_tweet = {
            'tokenize_text': word_tokenize(tweet.get('cleansing_text')),
            'original_tokenize' : tweet.get('cleansing_text')
        }
        tokens.append(modified_tweet)
        
    for tweet in tokens:
        modified_tweet = {
            'stopwords_text': remove_stopwords(tweet.get('tokenize_text')),
            'original_stopwords' :tweet.get('tokenize_text')
        }
        stopwords_list.append(modified_tweet)
        
    for tweet in stopwords_list:
        modified_tweet = {
            'stemming_text': [stemmer.stem(token) for token in tweet.get('stopwords_text')],
            'original_stemming' : tweet.get('stopwords_text')
        }
        stemmed_tokens.append(modified_tweet)
        
    for tweet in stemmed_tokens:
        modified_tweet = {
            'join_text': ' '.join(tweet.get('stemming_text')),
            'original_jointext' : tweet.get('stemming_text')
        }
        stemmed_text.append(modified_tweet)
        
    for tweet in stemmed_text:
        modified_tweet = {
            'slangremove_andfinaltext' : slang_remove(tweet.get('join_text')),
            'original_slangandfinaltext' : tweet.get('join_text')
        }
        slang_removed_text.append(modified_tweet)
        
    for text_original, slang_tweet in zip(text_lower_case, slang_removed_text):
        modified_tweet = {
            'original_casefolding': text_original.get('original_casefolding'),
            'slangremove_andfinaltext': slang_tweet.get('slangremove_andfinaltext')
        }
        final_process.append(modified_tweet)
    
    return jsonify({
        'remove_emoji' : remove_emoji,
        'remove_character': remove_character,
        'text_lower_case': text_lower_case,
        'tokens': tokens,
        'stopwords_list': stopwords_list,
        'stemmed_tokens': stemmed_tokens,
        'stemmed_text': stemmed_text,
        'slang_remove' : slang_removed_text,
        'final_process': final_process
    })
################################################################



@app.route('/processing', methods=['POST'])
def process_data():
    json_data = request.get_json()
    data = json_data.get('data')
    ratio = json_data.get('ratio')
    test_samples = json_data.get('test_samples')
    random_state = json_data.get('random_state')

    processed_texts = []
    labels = []
    list_emosi = []
    positive_emotions_list = []
    negative_emotions_list = []
    label_encoder.fit([entry.get('Emosi') for entry in data])


    for entry in data:
        tweet_text = entry.get('Tweet Text')
        emosi = entry.get('Emosi')
        label = entry.get('Label')
    # Preprocess the tweet text
        processed_texts.append(tweet_text)
        list_emosi.append(emosi)
        emosi_encoded = label_encoder.transform([emosi])[0]
        
        positive_emotion = 1 if label_encoder.inverse_transform([emosi_encoded])[0] in positive_emotions else 0
        negative_emotion = 1 if label_encoder.inverse_transform([emosi_encoded])[0] in negative_emotions else 0


        positive_emotions_list.append(positive_emotion)
        negative_emotions_list.append(negative_emotion)
        labels.append(label)

# Create a DataFrame from the processed texts and emotion features
    df = pd.DataFrame({'Processed Text': processed_texts,'Emosi' :list_emosi, 'Label': labels})
    # Print the 'Emosi' column
    print(df['Emosi'])


# Apply conversion to positive and negative emotions using lambda functions
    df['Positive Emotion'] = positive_emotions_list
    df['Negative Emotion'] = negative_emotions_list
    # Return the DataFrame
       
    # Vectorize the preprocessed text
    # Create a regex tokenizer
    tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
    vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize, stop_words=stop_words,ngram_range=(1, 1))
    text_features = vectorizer.fit_transform(df['Processed Text'])

    # Convert emotion flags to numpy arrays
    positive_emotions_array = np.array(df['Positive Emotion']).reshape(-1, 1)
    negative_emotions_array = np.array(df['Negative Emotion']).reshape(-1, 1)

    # Concatenate the vectorized text features and the emotion features
    concatenated_features = hstack((text_features, positive_emotions_array, negative_emotions_array))

    # Assign the concatenated features to the dataframe
    df_concatenated = pd.DataFrame(concatenated_features.toarray())
    df_concatenated['Label'] = df['Label']

    # Encode the labels
    y = label_encoder.fit_transform(df_concatenated['Label'])

    
    if ratio is not None and test_samples is not None:
        return jsonify({'error': 'Please provide either "ratio" or "test_samples", not both.'})

    if ratio is not None:
        X_train, X_test, y_train, y_test = train_test_split(df_concatenated.drop('Label', axis=1), y, test_size=ratio, random_state=random_state)
    elif test_samples is not None:
        if test_samples >= len(df_concatenated):
            return jsonify({'error': 'The number of test samples cannot exceed the total number of samples.'})
        X_train, X_test, y_train, y_test = train_test_split(df_concatenated.drop('Label', axis=1), y, test_size=test_samples, random_state=random_state)
    else:
        return jsonify({'error': 'Please provide either "ratio" or "test_samples" parameter.'})

    # Split the data into training and testing sets
    X_train.columns = X_train.columns.astype(str)
    X_test.columns = X_test.columns.astype(str)

    #get total number of data
    total_data = len(df_concatenated)
    #get number of sample for training
    num_train_samples = len(X_train)
    #get number of sample for testing
    num_test_samples =len(X_test)
    

    # Apply Laplace smoothing
    alpha = 2.0  # Laplace smoothing parameter

    # Train a Naive Bayes classifier with Laplace smoothing
    clf = MultinomialNB(alpha=alpha)
    clf.fit(X_train, y_train)
    


    # Make predictions on the test set
    y_pred = clf.predict(X_test)


# Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)



    response = {
        'total_data':total_data,
        'num_train_samples':num_train_samples,
        'num_test_samples' : num_test_samples,
        'predictions': [],
        'training' : [],
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'ratio' : ratio
    }

    for entry, predicted_label, label in zip(X_test.index, y_pred, df['Label']):
        tweet_text = df['Processed Text'][entry]
        emosi = df['Emosi'][entry]
        predicted_label = label_encoder.inverse_transform([predicted_label])[0]

        response['predictions'].append({
        'Tweet_Text': tweet_text,
        'Emosi': emosi,
        'Label': label,
        'Predicted_Label': predicted_label
    })
    
    # Add training data
    for entry, label in zip(X_train.index, df['Label']):
        tweet_text = df['Processed Text'][entry]
        emosi = df['Emosi'][entry]

        response['training'].append({
        'Tweet_Text': tweet_text,
        'Emosi': emosi,
        'Label': label
    })




    return jsonify(response)

if __name__ == '__main__':
    app.run()







################################################################

@app.route('/prediction', methods=['POST'])
def predict_dataset():
    try:
        # Extract the 'data' field from the JSON payload as an array
        texts = request.json['data']

        # Perform text classification for each text in the array
        predictions = []
        true_labels = []
        for item in texts:
            text = item['join_text'].lower()  # Extract the text from the 'join_text' field

            text_counts = vectorizer_dataset.transform([text])
            predicted_label = model_dataset.predict(text_counts)[0]
            predicted_label = int(predicted_label)
            predictions.append({'sentiment': predicted_label, 'text': text})
            true_labels.append(0)  # Assuming all ground truth labels are 0 for this example
        
        # Calculate evaluation metrics
        predicted_labels = [pred['sentiment'] for pred in predictions]
        confusion = confusion_matrix(true_labels, predicted_labels)
        accuracy = accuracy_score(true_labels, predicted_labels)
        precision = precision_score(true_labels, predicted_labels)
        recall = recall_score(true_labels, predicted_labels)

        # Create the response dictionary including predictions and evaluation metrics
        response = {
            'predictions': predictions,
            'confusion_matrix': confusion.tolist(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

        # Return the response dictionary as JSON response
        return jsonify(response)
    
    except Exception as e:
        # Return an error message if an exception occurs
        return jsonify({'error': str(e)})
    

if __name__ == '__main__':
    app.debug = True
    app.run()
