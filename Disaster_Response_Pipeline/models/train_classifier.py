# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
import time
import re
import sys
import os
from sqlalchemy import create_engine
import pickle

#import nltk 
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
nltk.download(['punkt', 'wordnet', 'stopwords'])

#import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
import warnings
warnings.filterwarnings('ignore')


def load_data(database_filepath):
    """
    Loads the SQLite database from the given database_filepath. Divide the
    data into model inputs and message labels.
    
    INPUTS:
        database_filepath - path to the SQLite database containing the messages
    RETURNS:
        X - inputs to be used for modeling. Contains the messages.
        y - labels for modeling. Contains the categories of the messages
        category_names - list conatining all types of message categories
    """
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Response', engine)
    categories = df.columns[4:]
    X = df['message']
    y = df.iloc[:, 4:]

    return X, y, categories 



def tokenize(text):
    """
    Clean and tokenize text for modeling. It will replace all non-
    numbers and non-alphabets with a blank space. Next, it will
    split the sentence into word tokens and lemmatized them with Nltk's 
    WordNetLemmatizer(), first using noun as part of speech, then verb.
    
    INPUTS:
        text - the message to be clean and tokenized
    RETURNS:
        words: the list containing the cleaned and tokenized words of
                        the message
    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # Detecte URLs
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    words = word_tokenize(text)
    
    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]
    
    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]
    
    return words


def build_model():
    """
    Builds the pipeline that will transform the messages and the model them
    based on the user's model selection. It will also perform a grid search
    to find the optimal model parameters.
    
    INPUT:
        model_type - the model type selected by the user.
    :RETURN: model - GridSearchCV object
    """
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultiOutputClassifier(
                            OneVsRestClassifier(LinearSVC())))])

    # hyper-parameter grid
    parameters = {'vect__ngram_range': ((1, 1), (1, 2)),
                  'vect__max_df': (0.75, 1.0)
                  }

    # create model
    model = GridSearchCV(estimator=pipeline,
            param_grid=parameters,
            verbose=3,
            cv=3, n_jobs= -1)
    return model


def evaluate_model(model, X_test, y_test, categories):
    """
    Evaluate the model's performance on the test data. Will return the
    precision, recall and f1 score for each category.
    
    INPUTS:
        model - the optimized model used to classify messages
        X_test - the model inputs of the test data
        y_test - the labels of the messages from the test data
        category_names - the names of all message categories
    """
    #category_names    = y.columns
    y_pred = model.predict(X_test)
    for i, column in enumerate(categories):
        print(classification_report(y_test[column], y_pred[:, i]))


def save_model(model, model_filepath):
    """
    Save the optimized model to the path specified by model_filepath
    
    INPUTS:
        model - the optimized model
        model_filepath - the path where the model will be saved
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file, -1)          


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, categories = load_data(database_filepath)
        X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, categories)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
