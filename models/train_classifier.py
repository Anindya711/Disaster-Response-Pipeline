#imporing important libraries
import sys
#importing text processing libraries from nltk
import nltk
nltk.download(['wordnet', 'punkt', 'stopwords'])

# import general python libraries
import numpy as np
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
from nltk import word_tokenize, sent_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from nltk.corpus import stopwords
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.externals import joblib
from sklearn.preprocessing import StandardScaler
#importing stopwords to get a list of stopwords
stopWords = set(stopwords.words('english'))

#import warning to ignore printing them
import warnings
warnings.filterwarnings("ignore")
from custom_transformer import AverageWordLengthExtractor

def load_data(database_filepath):
    # load data from database

    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterMessages', engine)
    df.related = df.related.replace(2,1)

    #Splitting the data to get X, y values
    #Our feature here is the message text, as we are only interested in classifying the text
    X = df.message
    #all the categories will be our target as this is a multiclass classification problem
    y = df[[col for col in df.columns if col not in  ['message', 'id', 'original','genre']]]
    categories = y.columns
    return X, y, categories

def tokenize(text):

    #tokenize text using word tokenize
    tokens = word_tokenize(text)

    #initiate the lemmatizer
    lemmatizer= nltk.wordnet.WordNetLemmatizer()

    clean_tokens=[]
    stopwords=[]
    #normalize case, lemmatize, strip leading/trailing whitespaces from the tokens
    for token in tokens:
        #checking if the token is a stopword
        if token in stopWords:
            stopwords.append(token)
        else:
            token=token.lower().strip()
            token=lemmatizer.lemmatize(token)
            clean_tokens.append(token)
    return clean_tokens



def build_model():
    # text processing and model pipeline

    '''
    model = Pipeline([('vect' ,  CountVectorizer(tokenizer=tokenize)),
                     ('tfidf' , TfidfTransformer()),
                     ('clf' , MultiOutputClassifier(RandomForestClassifier()))
                    ])
    parameters = {
                    'vect__max_df': [0.9, 0.95],
                    'vect__ngram_range': [(1,1), (1,2)],
                    'clf__estimator__n_estimators' : [50,70]
                    #'clf__estimator__min_samples_leaf': [1,2],
                    #'clf__estimator__min_samples_split': [2,3,4]
             }
    '''
    vect = Pipeline([('vect', CountVectorizer(tokenizer=tokenize, min_df=.0025, max_df=0.25, ngram_range=(1,3))),
                     ('tfidf', TfidfTransformer())
                    ])
    #print(vect.fit_transform(X))
    text_length = Pipeline([('text_length', AverageWordLengthExtractor()),
                          ('scale', StandardScaler())
                         ])
    #print(np.any(np.isnan(X)))
    #print(text_length.fit_transform(X[:100]))
    feats = FeatureUnion([('vect', vect), ('text_length', text_length)])
    feature_processing = Pipeline([('feats', feats)])
    #feature_processing.fit_transform(X_train)
    model = Pipeline([
                           ('features',feats),
                           ('classifier', MultiOutputClassifier(AdaBoostClassifier(random_state = 42)))
                        ])
    #print(model.get_params().keys())
    parameters = {
                   #'features__vect__vect__ngram_range'  : [(1,1), (1,2)],
                   'classifier__estimator__n_estimators' : [10,100,200],
                   'classifier__estimator__learning_rate' : [0.01,0.03,0.05,0.07,0.1,0.2,0.5,1]

                 }

    model = GridSearchCV(model, param_grid=parameters, verbose=2, n_jobs = -1)


    return model


def evaluate_model(model, X_test, Y_test, category_names):

    Y_pred = model.predict(X_test)

    print(classification_report(Y_test, Y_pred, target_names=category_names))



def save_model(model, model_filepath):
    # Save best grid search pipeline to file
    dump_file = model_filepath
    joblib.dump(model, dump_file, compress=1)


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

        print("Best Parameters : ")
        print(model.best_params_)

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
