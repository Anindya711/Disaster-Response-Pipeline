import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
pd.set_option('display.max_columns', 999)

def load_data(messages_filepath, categories_filepath):
    '''
        This function will take the path of the message and categories files
        and load them into two pandas dataframe and then merge them into a
        single dataframe. This function will return the merged dataframe.
        input : path to the source files (str)
        return : merged dataframe, pandas object

    '''
    #loading the datasets into different dataframes
    #print(messages_filepath, categories_filepath)
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    #merging the dataframes into a single dataframe
    merged_data= pd.merge(messages, categories, how='inner', on='id')

    #check results
    #print("First 5 rows of the merged dataset")
    #print(merged_data.head())

    #return the merged dataframe
    return merged_data
def clean_data(df):
	'''
		This function will take as input a pandas dataframe and apply several
		cleaning operations to it and finally will return the cleaned dataframe.
		input : dataframe, pandas object
		return : cleaned dataframe, pandas object
	'''
	#Split categories into separate category columns.
	#create a dataframe of the 36 individual category columns
	categories = df.categories.str.split(';', expand=True)

	#select the first row of the categories dataframe
	row = categories.loc[0]

	#extract the text fields to be used as category column names
	category_colnames = row.apply(lambda x : x[:-2])

	#rename the column names for categories dataframe
	categories.columns = category_colnames

	#Convert the category values to just numbers 0 and 1
	for column in categories.columns:
		categories[column] = categories[column].str[-1]
		#convert the column values from string to numeric
		categories[column]=pd.to_numeric(categories[column])

	#drop the categories column from the dataframe df
	df=df.drop('categories', axis = 1)

	#concat the categories dataframe to the original dataframe so that all the
	#categories are present in the final dataframe for analysis
	df_cleaned=pd.concat([df, categories], axis=1)

	#drop any duplicates in the dataset
	df_cleaned.drop_duplicates(inplace= True)

	#return the cleaned dataframe
	return df_cleaned

def save_data(df, database_filepath):
	'''
		This function will take as input a pandas dataframe and save it to a
		sqllite database.

	'''
	engine = create_engine('sqlite:///'+database_filepath)
	df.to_sql('DisasterMessages', engine, index=False)

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath= sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':

    main()
