import pandas as pd
import numpy as np

fake = pd.read_csv("fake.csv")
true = pd.read_csv("True.csv")
true.head()

true["lable"] = 1
fake["lable"] = 0
true.head()

news = pd.concat([true, fake], axis=0)

news = news.drop(columns="title")
news = news.drop(columns="subject")
news = news.drop(columns="date")

news.head()

import re

def wordopt(text):
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r'[^\\w\\s]', "", text)
    text = re.sub(r'<.*?>', "", text)
    text = re.sub(r"\d", "", text)
    text = re.sub(r"\n", " ", text)
    return text

news["text"] = news["text"].apply(wordopt)

news["text"]

x = news['text']
y = news["lable"]

x

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

x_train.shape

from sklearn.feature_extraction.text import TfidfVectorizer

vectorzation = TfidfVectorizer()

xv_train = vectorzation.fit_transform(x_train)
xv_test = vectorzation.transform(x_test)

xv_test

from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(xv_train, y_train)

pred_lr = LR.predict(xv_test)

LR.score(xv_test, y_test)

from sklearn.tree import DecisionTreeClassifier

DTC = DecisionTreeClassifier()

DTC.fit(xv_train, y_train)

pred_dtc = DTC.predict(xv_test)

DTC.score(xv_test, y_test)

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()

rfc.fit(xv_train, y_train)

predict_rfc = rfc.predict(xv_test)

rfc.score(xv_test, y_test)

def output(n):
    if n == 0:
        return "FAKE NEWS"
    elif n == 1:
        return "GENUINE NEWS"

def manualtesting(news):
    print("Testing input:", news)
    testing_news = {"text": news}
    new_det_test = pd.DataFrame([testing_news])
    new_def_test = new_det_test.copy()
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    print("Processed text:", new_def_test['text'].iloc[0])
    
    new_x_test = new_def_test["text"]
    new_xv_test = vectorzation.transform(new_x_test)
    
    pred_lr = LR.predict(new_xv_test)
    pred_rfc = rfc.predict(new_xv_test)
    
    return "LR Prediction: {} RFC Prediction: {}".format(output(pred_lr[0]), output(pred_rfc[0]))

news_article = str(input("Enter the news article you want to test: "))
print(manualtesting(news_article))
import gradio as gr

def detect_fake_news(news_article):
    return manualtesting(news_article)

iface = gr.Interface(
    fn=detect_fake_news, 
    inputs=gr.Textbox(lines=5, placeholder="Enter a news article..."), 
    outputs="text",
    title="Fake News Detector",
    description="Enter a news article and check if it's FAKE or GENUINE."
)

iface.launch()