**About the Project** \
In this project I am going to be analyze thousands of real messages provided by **Figure 8**, sent during natural disasters either via social media or directly to disaster response organizations. I will build an ETL pipeline that processes message and category data from csv files and load them into a SQLite database, which the machine learning pipeline will then read from to create and save a multi-output supervised learning model. Then, I will create one web app which will extract data from this database to provide data visualizations and classify new messages for 36 categories.

Machine learning is critical to helping different organizations understand which messages are relevant to them and which messages to prioritize. During these disasters is when they have the least capacity to filter out messages that matter, and find basic methods such as using key word searches to provide trivial results. In this course, you'll learn the skills you need in ETL pipelines, natural language processing, and machine learning pipelines to create an amazing project with real world significance.

**Structure** \
The repository contains following folders.
*1. app* - contains the files required to run the web app.\
*2. models* - the trained machine learning model.\
*3. data* - contains the data files required to run the the ETL pipeline.

There are three components in this project.

**1. ETL Pipeline** \
The *process.py* script in data module will load the messages and categories datasets and then,
>1.Merges the two datasets.\
>2.Cleans the data.\
>3.Stores it in a SQLite database.

**2. ML Pipeline** \
The Python script, *train_classifier.py*,will do the following steps :
>1.Loads data from the SQLite database.\
>2.Splits the dataset into training and test sets.\
>3.Builds a text processing and machine learning pipeline.\
>4.Trains and tunes a model using GridSearchCV.\
>5.Outputs results on the test set.\
>6.Exports the final model as a pickle file.

**3. Flask Web App**
>The results are then displayed via a web app built on flask. The user will enter a message and then 36 possible outputs are displayed
on the screen.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

