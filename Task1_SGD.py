#Import Libraries
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

#Fetch data and print it
mnist = fetch_openml("mnist_784")

#Print the shape of the images
print("Image Data Shape" , mnist.data.shape)
x = mnist.data

#Print the shape of the labels
y= mnist.target
print("Label Data Shape", mnist.target.shape)

#Split the data to train images and labels , test images and labels 
y = y.astype(np.uint8)
train_img,test_img,train_lbl, test_lbl = train_test_split(x,y, test_size=0.1, random_state=42)
print(train_img.shape)
print(test_img.shape)
print(train_lbl.shape)
print(test_lbl.shape)

#Train our model to predict value 3
train_lbl_3 = (train_lbl == 3)
test_lbl_3 = (test_lbl == 3)

#Using SGDClassifier to make Binary Classification
sgd_binaryclassification = SGDClassifier(max_iter=100, random_state=42)
sgd_binaryclassification.fit(train_img, train_lbl_3)
predicted_binary=sgd_binaryclassification.predict(test_img)
print(predicted_binary)

#Confusion Matrix
conf_matrix = confusion_matrix(test_lbl_3, predicted_binary)
print(conf_matrix)
accuracy = accuracy_score(test_lbl_3,predicted_binary)
print("Accuracy",accuracy)

# Using SGDClassifier to make Multi Classes
sgd_multiclasses=SGDClassifier(max_iter=100, random_state=0)
sgd_multiclasses.fit(train_img,train_lbl)
predicted_classes=sgd_multiclasses.predict(test_img)
print(predicted_classes)
print(predicted_classes.shape)


