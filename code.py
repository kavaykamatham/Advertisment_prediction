import pandas as pd
import numpy as np
data = pd.read_csv('/content/drive/MyDrive/FLASK ML/Ads/ad_10000records.csv')
data
data.info()
data = data.drop(['Ad Topic Line', 'City', 'Country', 'Timestamp'], axis=1)
data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
data
data["Clicked on Ad"].value_counts()
click_through_rate = 4917 / 10000 * 100
print(click_through_rate)
x = data.drop(['Clicked on Ad'], axis = 1)
y = data['Clicked on Ad']
!pip install pycaret
from pycaret.classification import *
s = setup(data = data, target = 'Clicked on Ad', session_id=123)
best_model = compare_models()
model = create_model('rf')
model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=4)
model.fit(x_train, y_train)
x_train_prediction = model.predict(x_train)
x_train_prediction
x_test_prediction = model.predict(x_test)
x_test_prediction
training_data_accuracy = accuracy_score(x_train_prediction, y_train)

test_data_accuracy = accuracy_score(x_test_prediction, y_test)
print('Accuracy on Training data : ', training_data_accuracy)

print('Accuracy score on Test Data : ', test_data_accuracy)
print("Ads Click Through Rate Prediction : ")
a = float(input("Daily Time Spent on Site: "))
b = float(input("Age: "))
c = float(input("Area Income: "))
d = float(input("Daily Internet Usage: "))
e = input("Gender (Male = 1, Female = 0) : ")

features = np.array([[a, b, c, d, e]])
prediction = model.predict(features)

if prediction[0] == 1:
  print('User will click on Ad')

else:
  print('User will not click on Ad')
import pickle

with open('ad_prediction_model.pkl', 'wb') as file:
    pickle.dump(model, file)
    