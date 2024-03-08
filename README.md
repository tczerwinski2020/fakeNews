# fakeNews

Three programs to detect fake news. The first is through TFIDF and PAC. Second is TFIDF and Logistical regression. The third was through a count vectorizer and Multinomial Naive Bayes classifier. This program was written in Python. The training data set was too large of a file to upload. 


Abstract

We were given a classification problem to determine whether an article was fake news or real news. To solve this, we built three fake news detection models in Python, each based on different data analysis methods:
TF-IDF and Passive Aggressive Classifier
TF-IDF and Logistical Regression
CountVectorizer and Multinomial Naive Bayes Classifier
Each of these models has two components, a feature constructor and a classifier. Feature constructors take articles and process it into data that is easier to interpret and classifiers take that data and predict something from it. 


Background

With the rise of online journalism through alternative media outlets and social media, the issue of fake news has become significantly more prevalent in mainstream discourse. The best way to combat the consumption of fake news is to verify sources manual using basic source checking techniques like the ones seen below:

That being said, the volume of news media being spread on the internet each day is simply too high to manually account for every article, tweet, and blog post, necessitating an autonomous solution to the problem. Machine learning has been the backbone to a variety of algorithmic approaches to detecting and classifying fake news. These ML implementations involve training a model on a large number of articles, which then constructs an algorithm to predict the validity of future articles passed through the model. These models usually involve two key components,  feature construction and classification. Feature Construction is the concept of taking the set of raw inputs (in this case, articles, tweets, etc) and reducing them to some set of “features” that can then be easily interpreted by a classifier. It is then the job of the classifier to use the prior data its been fed, to make a prediction about how that piece of data should be grouped (in our case, whether it is fake or real news). 


Methods

Through our three fake news detectors, we used a multitude of machine learning and natural language processing tools. These include TF-IDF, Passive Aggressive Classifier, logistic regression modeling, count vectorizer, and the Multinomial Naive Bayes (MNB) algorithm. 

Cleaning the Data
To clean the data, we used a combination of techniques. First, we removed extra white spaces on the column names. Next, we converted numerical data into categorical data (REAL/FAKE). Third, we replaced nulls with the empty string. Lastly, we removed links from the data and everything that isn’t a character or punctuation. We then removed stop words and put all string data to lowercase. 

TF-IDF
Term Frequency-Inverse Document Frequency (TF-IDF) is a feature construction method (or feature vectorizer to be more precise). It seeks to determine the relevance of terms to an article by measuring their frequency in relation to the size of the article. TF-IDF is often used in information retrieval and machine learning.

Count Vectorizer
The Count Vectorizer is a feature vectorizer similar to the TF-IDF. Rather than trying to calculate the relevance of terms, it simply stores the number of times each term appears in the text within a sparse matrix. 

Passive Aggressive Classifier
Sklearn’s Passive Aggressive Classifier is an algorithm often used in large-scale machine learning. It classifies something as passive if the model fits, and aggressive if the model needs to be changed. 

Logistical Regression
Logistical Regression is in a popular data analysis tool used to find the relationship between two factors in data. In our case, we would be looking at the relationship between the term relevance values found by our TF-IDF and the validity of the news article. 

Multinomial Naive Bayes
Multinomial Naive Bayes is an algorithm used for text classification. Used in Natural Language Processing (NLP) problems, it is particularly useful for problems that involve text counts. 


Results

Our TF-IDF and PAC program was found to be 97% accurate. In the test data, 2611 fake articles were found and 2589 real ones. Otherwise, 50.21% of the articles in the test data were fake according to this model (49.79% were identified as real articles). The TF-IDF and LR algorithm was found to be 94.88% accurate, finding 2609 fake articles and 2591 real articles in the test data. This program identified 50.17% as fake articles and 49.83% as real articles. Lastly, our Count Vectorizer and Multinomial NB program was found to be 91.32% accurate, with 2963 fake articles identified and 2237 real ones. This program found 56.98% of the test data to be fake articles and 43.02% 


#
Accuracy

Term Frequency-Inverse Document Frequency (TF-IDF)

Passive Aggressive Classifier 
97.00 %

Term Frequency-Inverse Document Frequency (TF-IDF)

Logistical Regression
94.88 %

CountVectorizer

Multinomial Naive Bayes Classifier
91.32 %


#

Number of Test Articles Found to be Fake

Number of Test Articles Found to be Real

Term Frequency-Inverse Document Frequency (TF-IDF)
Passive Aggressive Classifier 

2611

2589

Term Frequency-Inverse Document Frequency (TF-IDF)
Logistical Regression

2609

2591

CountVectorizer=
Multinomial Naive Bayes Classifier

2963

2237









Sources
Spotting Fake News Infographic
TF-IDF + PassiveAggressiveClassifier Implementation
TF-IDF + Logistical Regression Implementation

