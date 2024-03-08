import numpy as np
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier 
import nltk
import regex as re
from nltk.stem.porter import PorterStemmer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt


nltk.download('stopwords')
nltk.download('wordnet')
ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()

stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)



# Cleaning text from unused characters
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation
    text = str(text).replace('[^a-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    #text = ' '.join(text)    
    return text

## Nltk Preprocessing include:
# Stop words, Stemming and Lemmetization
# For our project we use only Stop word removal
def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    #text = ' '.join([word for word in wordlist if word not in stopwords_dict])
    #text = [ps.stem(word) for word in wordlist if not word in stopwords_dict]
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return  text





def clean_data(df):
	df = convert_num(df)
	for col in df.columns:
		df[col] = df[col].astype('str')
		df = remove_nan(df, col)
		df = remove_col_white_space(df, col)
	return df


def convert_num(df):
    # Convert categorical variable to numerical variable
    num_encode = {'label'  : {1:'REAL', 0:'FAKE'}  }
    df.replace(num_encode, inplace=True)  
    return df

def remove_nan(feature_df, col):
    for col in ['text', 'author', 'title']:
        feature_df.loc[feature_df[col].isnull(), col] = ""
    return feature_df


def remove_col_white_space(df,col):
    # remove white space at the beginning of string 
    df[col] = df[col].str.lstrip()
    df[col] = df[col].str.rstrip()
    return df


def tfidf_pac(x_train, y_train, x_test):
#https://data-flair.training/blogs/advanced-python-project-detecting-fake-news/
	hold = x_test
	print('TFIDF and PAC')
	tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)
	x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.2, stratify=y_train, random_state=2)

	#DataFlair - Fit and transform train set, transform test set
	tfidf_train = tfidf_vectorizer.fit_transform(x_train)

	tfidf_test = tfidf_vectorizer.transform(x_test)

	#DataFlair - Initialize a PassiveAggressiveClassifier
	pac=PassiveAggressiveClassifier(max_iter=50)
	pac = pac.fit(tfidf_train, y_train)

	#DataFlair - Predict on the test set and calculate accuracy
	y_pred = pac.predict(tfidf_test)

	#training data
	training_data_accuracy = accuracy_score(y_pred, y_test)
	print('Accuracy score of the TRAINING data : ', training_data_accuracy)

	a = np.array(y_pred)
	a = pd.DataFrame(a)
	print(a.value_counts())
	print(a.value_counts()/len(x_test))
	print()


	#test data
	test = tfidf_vectorizer.transform(hold)
	test_pred = pac.predict(test)
	a = np.array(test_pred)
	a = pd.DataFrame(a)
	print("Results of the TEST data: ")
	print(a.value_counts())
	print(a.value_counts()/len(hold))

	print("END TFIDF and PAC")
	print()

	df = pd.DataFrame(y_pred, columns=['RorF'])
	sns.countplot(df['RorF'])
	plt.title('Number of Test Articles Access to be Real or Fake by TFIDF and Passive Aggressive Classifier')
	plt.xlabel('No. of Articles')
	plt.ylabel('Real or Fake')
	plt.show()
	return y_pred


def stemming(content):
	stemmed_content = re.sub('[^a-zA-Z]',' ',content)
	stemmed_content = stemmed_content.lower()
	stemmed_content = stemmed_content.split()
   # stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
	stemmed_content = ' '.join(stemmed_content)
	return stemmed_content

def tfidf_log_reg(x_train, y_train, x_test):
	#https://ai.plainenglish.io/fake-news-detection-project-using-machine-learning-explained-with-code-8f83ae5f7a26
	print("TFIDF LOG REG")

	port_stem = PorterStemmer()
	x_train = x_train.apply(stemming)
	X = x_train.values
	Y = y_train.values
	vectorizer = TfidfVectorizer()
	vectorizer.fit(X)

	X = vectorizer.transform(X)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
	model = LogisticRegression()
	model.fit(X_train, Y_train)
	X_train_prediction = model.predict(X_train)
	training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
	print('Accuracy score of the training data : ', training_data_accuracy)

	X_test_prediction = model.predict(X_test)
	test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
	print('Accuracy score of the training test data : ', test_data_accuracy)


	print('Test Data Results: ')
	test_data = vectorizer.transform(x_test.values)
	y_prediction_test = model.predict(test_data)
	a = np.array(y_prediction_test)
	a = pd.DataFrame(a)
	print(a.value_counts())
	print(a.value_counts()/len(x_test))


	df = pd.DataFrame(y_prediction_test, columns=['RorF'])
	sns.countplot(df['RorF'])
	plt.title('Number of Test Articles Access to be Real or Fake by TFIDF and Logistic Regression')
	plt.xlabel('No. of Articles')
	plt.ylabel('Real or Fake')
	plt.show()

	print("END TFIDF LOG REG")


def count_vec(x_train, y_train, x_test):
	#https://heartbeat.comet.ml/fake-news-detection-with-python-d7339cf1f018
	print()
	print("START COUNT VEC")
	cv = CountVectorizer()
	x = cv.fit_transform(x_train)
	xtr, xte, ytr, yte = train_test_split(x, y_train, test_size = 0.2, random_state = 2)
	model = MultinomialNB()
	model.fit(xtr, ytr)
	X_test_prediction = model.predict(xte)
	print("training test data")
	training_data_accuracy = accuracy_score(X_test_prediction, yte)
	print('Accuracy score of the training data : ', training_data_accuracy)

	print('Test Data Results: ')
	test_data = cv.transform(x_test.values)
	y_prediction_test = model.predict(test_data)
	a = np.array(y_prediction_test)
	a = pd.DataFrame(a)
	print(a.value_counts())
	print(a.value_counts()/len(x_test))


	df = pd.DataFrame(y_prediction_test, columns=['RorF'])
	sns.countplot(df['RorF'])
	plt.title('Number of Test Articles Access to be Real or Fake by Count Vectorizer and MultinomialNB')
	plt.xlabel('No. of Articles')
	plt.ylabel('Real or Fake')
	plt.show()

	print("END COUNT VEC")





def main():
	#Read the data
	df=pd.read_csv('Project 1 dataset/train.csv')
	df_test = pd.read_csv('Project 1 dataset/test.csv')

	#clean the data
	df = clean_data(df)
	df_test = clean_data(df_test)

# apply preprocessing on text through apply method by calling the function nltk_preprocess
	df["text"] = df.text.apply(nltk_preprocess)
# apply preprocessing on title through apply method by calling the function nltk_preprocess
	df["title"] = df.title.apply(nltk_preprocess)
	df['author'] = df.author.apply(nltk_preprocess)
	df_test["text"] = df_test.text.apply(nltk_preprocess)
# apply preprocessing on title through apply method by calling the function nltk_preprocess
	df_test["title"] = df_test.title.apply(nltk_preprocess)
	df_test['author'] = df_test.author.apply(nltk_preprocess)



	df['content'] = df['author']+' '+df['title'] +' '+df['text']
	X = df.drop(columns='label', axis=1)
	X = X.drop(columns='title', axis=1)
	X = X.drop(columns='author', axis=1)
	X = X.drop(columns='text', axis=1)
	df_test['content'] = df_test['author']+' '+df_test['title'] +' ' +df_test['text']
	df_test = df_test.drop(columns='title', axis =1)
	df_test = df_test.drop(columns='author', axis =1)
	df_test = df_test.drop(columns='text', axis =1)
	df_test['id'] = df_test['id'].astype("string")

	#training data
	x_train = X['content']
	#test data
	x_test = df_test['content']
	#training results
	y_train = df['label']
	print(X)
	print(df_test)

	y_pred_tfidf_pac = tfidf_pac(x_train, y_train, x_test)


	y_pred_tfidf_log_reg = tfidf_log_reg(x_train, y_train, x_test)

	count_vec(x_train, y_train, x_test)


main()
