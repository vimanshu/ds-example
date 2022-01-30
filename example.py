import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)

knn = KNeighborsClassifier() 
knn.fit(X_train,y_train)

# knn_score = knn.score(X_test,y_test)

# save the weight in the pickle file
#  this is what i need to take away

#save
# with open('model.pkl','wb') as f:
#     pickle.dump(knn,f)

#load
# with open('model.pkl','rb') as f:
#     knn1 = pickle.load(f)


pickle.dump(knn, open('example_weights_knn.pkl', 'wb'))

