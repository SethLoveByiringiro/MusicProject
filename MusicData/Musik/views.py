import joblib
from django.shortcuts import render

import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Read CSV DATASET
data = pd.read_excel('/home/ezechielwill/PycharmProjects/MusicProject/files/musicdata.xlsx')

# Data Cleaning
# Since there is no empty cells and duplicated values

# Let's get our X & Y variables set

X = data.drop(columns=['S/N','DISTRICT NAME','ARTIST '])

y = data['ARTIST ']
# Split the training (70%) and testing (30%) variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

DT = joblib.load( '/home/ezechielwill/PycharmProjects/MusicProject/Decision.joblib')
SVM = joblib.load( '/home/ezechielwill/PycharmProjects/MusicProject/svm.joblib')
RF = joblib.load( '/home/ezechielwill/PycharmProjects/MusicProject/randomForest.joblib')
LR = joblib.load( '/home/ezechielwill/PycharmProjects/MusicProject/logistic.joblib')

MODELS = [DT, SVM, RF, LR]

def models_accuracy(DT, SVM, RF, LR):
    # predict for decision tree
    DTpred = DT.predict(X_test)
    scoreDT = accuracy_score(y_test, DTpred)
    # predict & check score for Support Vector M
    svmPred = SVM.predict(X_test)
    scoreSvm = accuracy_score(y_test, svmPred)
    # Predict & check sore for Random Forest
    RFpred = RF.predict(X_test)
    scoreRF = accuracy_score(y_test, RFpred)
    # predict & check score for Logistic Regression
    LRpred = LR.predict(X_test)
    scoreLR = accuracy_score(y_test, LRpred)

    return [scoreDT, scoreSvm, scoreRF, scoreLR]

def predictArtist(DT, user_input):
    DTpred = DT.predict(user_input)
    return DTpred

def displayAccuracy(request):
    models = models_accuracy(*MODELS)

    dt = models[0] * 100
    svm = models[1] * 100
    rf = models[2] * 100
    lr = models[3] * 100

    scores = {
        'Decision Tree' :round(dt),
        'Support Vector Machine': round(svm),
        'Random Forest': round(rf),
        'Logistic Regression': round(lr)
    }
    return render(request, 'popular.html', {'scores' : scores})
def display(request):

    if request.method == "POST":
        age = request.POST['age']
        gender = request.POST['gender']
        district_no = request.POST['district_no']

        input = [[age, gender, district_no]]


        result = predictArtist(DT, input)

        return render(request, 'popular.html', {'content': result[0]})
    models = models_accuracy(*MODELS)

    dt = models[0] * 100
    svm = models[1] * 100
    rf = models[2] * 100
    lr = models[3] * 100

    scores = {
        'Decision Tree': round(dt),
        'Support Vector Machine': round(svm),
        'Random Forest': round(rf),
        'Logistic Regression': round(lr)
    }
    return render(request, 'popular.html', {'scores': scores})
