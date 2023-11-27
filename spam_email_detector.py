import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/VANSH KHANEJA/Downloads/spam.csv")

data.head(10)

data.groupby('Category').count()

data.groupby('Category').count().plot(kind='pie',subplots=True)

data['spam'] = data['Category'].apply(lambda x: 1 if x=='spam' else 0)
data.head()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data.Message,data.spam)

from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,y_train)

email = input("Enter subject of email : ")

emails_count = v.transform([email])
ans = model.predict(emails_count)

if ans == [0]:
    print("Not a spam mail")
else:
    print("Its a spam mail")
    
