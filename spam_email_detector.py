import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("C:/Users/VANSH KHANEJA/Downloads/spam.csv")

data['spam'] = data['Category'].apply(lambda x: 1 if x=='spam' else 0)


v = CountVectorizer()
X_train_count = v.fit_transform(data.Message)

model = MultinomialNB()
model.fit(X_train_count,data.spam)

email = input("Enter subject of email : ")

def classify_mail(email):
    emails_count = v.transform([email])
    array_of_ans = model.predict(emails_count)
    for i in array_of_ans:
        if(i==1):
            return 'Its a Spam Mail'
        else:
            return 'Not a Spam Mail'
    
print(classify_mail(email))
