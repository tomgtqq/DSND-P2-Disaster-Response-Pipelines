import sys
import re
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'stopwords', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import pickle

def load_data(db_file_path):
    engine = create_engine('sqlite:///{}'.format(db_file_path))
    df = pd.read_sql_table('messages_table', engine)
    X = df.message
    Y = df.drop(['id','message','original','genre'], axis=1).fillna(0)
    return X, Y , Y.columns.tolist()

def tokenize(text):
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower() ) 
    # splite text into words
    words = word_tokenize(text)
    # remove stop words 
    words = [w for w in words if w not in stopwords.words("english")]
    # lemmatization word
    words = [WordNetLemmatizer().lemmatize(w).strip() for w in words]
    #stemming word
    words = [PorterStemmer().stem(w) for w in words]  
    return words

def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {  
         'clf__estimator__n_estimators': [20, 50], 
         'clf__estimator__min_samples_split': [4, 6]
    }
    cv = GridSearchCV(pipeline, param_grid=parameters, verbose = 4)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    # predict on test data
    Y_pred = model.predict(X_test)
    print('Categories of messages'.format(category_names))
    for i, col in enumerate(Y_test):
        print('Categories: {}'.format(col))
        print(classification_report(Y_test[col], Y_pred[:, i]))
    print('Accuracy: {}'.format((Y_test.values == Y_pred).mean()))


def save_model(model, model_filepath): 
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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