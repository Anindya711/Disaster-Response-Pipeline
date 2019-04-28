**About the Project**
In this project I am going to be analyze thousands of real messages provided by **Figure 8**, sent during natural disasters either via social media or directly to disaster response organizations. I will build an ETL pipeline that processes message and category data from csv files and load them into a SQLite database, which the machine learning pipeline will then read from to create and save a multi-output supervised learning model. Then, I will create one web app which will extract data from this database to provide data visualizations and classify new messages for 36 categories.

Machine learning is critical to helping different organizations understand which messages are relevant to them and which messages to prioritize. During these disasters is when they have the least capacity to filter out messages that matter, and find basic methods such as using key word searches to provide trivial results. In this course, you'll learn the skills you need in ETL pipelines, natural language processing, and machine learning pipelines to create an amazing project with real world significance.

There are three components in this project.

**1. ETL Pipeline**
The process.py script in data module will loads the messages and categories datasets and then
>1.Merges the two datasets
>2.Cleans the data
>3.Stores it in a SQLite database

**2. ML Pipeline**
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file
3. Flask Web App
We are providing much of the flask web app for you, but feel free to add extra features depending on your knowledge of flask, html, css and javascript. For this part, you'll need to:

Modify file paths for database and model as needed
Add data visualizations using Plotly in the web app. One example is provided for you

python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

python train_classifier.py ../data/DisasterResponse.db classifier.pkl
