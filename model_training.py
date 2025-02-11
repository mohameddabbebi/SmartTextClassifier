
import pandas as pd

import json

with open('/content/News_Category_Dataset.json', 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f if line.strip()]  # Read line by line

df = pd.DataFrame(data)[['headline', 'category']]  # Convert to DataFrame
print(df.head())
# Get the first 7 unique category values
unique_categories = df['category'].unique()[:8]
unique_categories[3]='SPORT'
print(unique_categories)
# Filter the dataframe to include only these categories
df_new = df[df['category'].isin(unique_categories)]

df_new.head()
df_balanced1 = pd.concat([df_balanced[x] for x in df_new['category']],axis=0)
df_balanced=df_balanced1
category_map = {cat: i for i, cat in enumerate(df_balanced['category'].unique())}
df_balanced['category_num'] = df_balanced['category'].map(category_map)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df_balanced.headline,df_balanced.category_num,test_size=0.2,random_state = 2022,stratify=df_balanced.category_num)

from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pipe
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
clf = Pipeline([('vectorize_bow',TfidfVectorizer()),('Multi NB',MultinomialNB())])
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(classification_report(y_test, y_pred))

#to save your model 
import joblib

joblib.dump(clf, 'news_classifier_model.pkl')

