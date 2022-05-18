#Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, time
warnings.filterwarnings('ignore')
import pickle
import nltk #Import NLTK ---> Natural Language Toolkit
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

class FakeNewsDetection():
    def __init__(self, input_file,test_data_ratio):
        self.df = pd.read_csv(input_file, encoding='utf-8',header=0)
        self.test_data_ratio = test_data_ratio
        print(self.df.head())
        print(self.test_data_ratio)
        
    #Preproccess the input data
    def preprocess_data(self,data):

      # 1. Tokenization
        tk = RegexpTokenizer('\s+', gaps = True)
        text_data = [] # List for storing the tokenized data
        for values in data.text:
            tokenized_data = tk.tokenize(values) # Tokenize the news
            text_data.append(tokenized_data) # append the tokenized data

      # 2. Stopword Removal
      # Extract the stopwords
        sw = stopwords.words('english')
        clean_data = [] # List for storing the clean text
      # Remove the stopwords using stopwords
        for data in text_data:
            clean_text = [words.lower() for words in data if words.lower() not in sw]
            clean_data.append(clean_text) # Appned the clean_text in the clean_data list

      # 3. Stemming
      # Create a stemmer object
        ps = PorterStemmer()
        stemmed_data = [] # List for storing the stemmed data
        for data in clean_data:
            stemmed_text = [ps.stem(words) for words in data] # Stem the words
            stemmed_data.append(stemmed_text) # Append the stemmed text


      # 4. tfidf vectorizer --> Term Frequency Inverse Document Frequency  
      # Flatten the stemmed data

        updated_data = []
        for data in stemmed_data:
            updated_data.append(" ".join(data))
        return updated_data
    
    def tfidf(self, data):
      # TFID Vector object
        self.tfidf = TfidfVectorizer()
        self.tfidf.fit(data)


    def compute_metrics(self,data, y_true, model_obj, model):

        # Make predictions
        y_pred = model_obj.predict(data)
        #Classification report and Confusion Matrix
        print(metrics.classification_report(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=model_obj.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=model_obj.classes_)
        disp.plot()
        plt.show()
    
    def detect(self):
        print("---Preprocessing data---")
        print(self.df.shape)
        self.df.drop_duplicates(inplace=True)
        print(self.df.shape)
        self.df.dropna(inplace=True)
        print(self.df.shape)
        preprocessed_data = self.preprocess_data(self.df.drop('label', axis=1))
        self.tfidf(preprocessed_data)
        preprocessed_data = self.tfidf.transform(preprocessed_data)
        print("shape of transform in training is: ", preprocessed_data.shape)
        features_df = pd.DataFrame(preprocessed_data.toarray())
        #label_df = pd.DataFrame(self.df.label)
        datadf = pd.concat([features_df,self.df.label],axis=1)

        print("Splitting data into train and test set")
        X_train, X_test, y_train, y_test = train_test_split(features_df, self.df.label, test_size=self.test_data_ratio, random_state=21)
        print("---Training the model on train set---")

        # Initialize the model
        mnb = MultinomialNB(alpha=0.1)

        # Fit the model
        mnb.fit(X_train, y_train)
        with open("fakenewsmodel.pkl","wb") as file:
            pickle.dump(mnb,file)
        #time.sleep(2)
        print("Naive Bayes model trained successfully!")
        print("Classification Metrics for the train set is:" + "\n")
        self.compute_metrics(X_train, y_train, mnb, 'Naive Bayes')
        #time.sleep(4)
        print("--- Testing the model on the test set---")
        print("Classification Metrics for the test set is:" + "\n")
        self.compute_metrics(X_test, y_test, mnb, 'Naive Bayes')
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

    def test_news(self, test_file):
        with open("fakenewsmodel.pkl", "rb") as file:
            model=pickle.load(file)
        df_test =pd.read_csv(test_file)
        #preprocessed_testdata = self.preprocess_data(df_test.drop('expected', axis=1))
        preprocessed_testdata = self.preprocess_data(df_test)
        preprocessed_testdata = self.tfidf.transform(preprocessed_testdata)
        print("shape of transform in testing is: ", preprocessed_testdata.shape)
        features_df = pd.DataFrame(preprocessed_testdata.toarray())

        print(features_df.shape)
        df_test["label"]=model.predict(features_df)
      
        print(model.predict_proba(features_df))
        print(model.classes_)
        probabs = model.predict_proba(features_df)

        probs = list()
        for prob in probabs:
            probs.append(round(max(prob[0],prob[1]), 2))
        df_test["probability"] = probs
        df_test.to_csv("test_pred.csv",index=False)
