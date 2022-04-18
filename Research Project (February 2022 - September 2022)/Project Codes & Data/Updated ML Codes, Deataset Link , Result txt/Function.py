# Dependencies
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#Preprocessing
def MinMax(xtrain,xtest):
    from sklearn.preprocessing import MinMaxScaler
    mmx = MinMaxScaler()
    xtrain = mmx.fit_transform(xtrain)
    xtest = mmx.transform(xtest)
    return xtrain, xtest


def Standard(xtrain,xtest):
    from sklearn.preprocessing import StandardScaler
    stdS = StandardScaler()
    xtrain = stdS.fit_transform(xtrain)
    xtest = stdS.transform(xtest)
    # print(f"Original Test set data ------> {}, Class")
    return xtrain, xtest


def LOF(xtrain, ytrain):
    from sklearn.neighbors import LocalOutlierFactor
    lof = LocalOutlierFactor(n_neighbors=15, novelty=False)
    # print(f"Original data are----> Features:{xtrain.shape}, Label:{ytrain.shape}")
    yhat = lof.fit_predict(xtrain)
    mask = yhat != -1
    xtrain, ytrain = xtrain[mask, :], ytrain[mask]
    # print(f"Outlierless data are----> Features:{xtrain.shape}, Label:{ytrain.shape}")
    return xtrain, ytrain


def OverSample (xtrain, ytrain):
    from imblearn.over_sampling import SMOTE
    oversample = SMOTE()
    xtrain, ytrain = oversample.fit_resample(xtrain, ytrain)
    # print(f"Shape after oversampling: {xtrain.shape, ytrain.shape}")
    return xtrain, ytrain


def ClassCount(ytrain):
    one_count, two_count, zero_count, others = 0,0,0,0
    for c in ytrain:
        if c == 1: 
            one_count+=1
        elif c == 0:
            zero_count += 1
        elif c == 2:
            two_count+=1
        else: 
            others+=1
    total = one_count+two_count+zero_count+others
    print(f"------> Two: {two_count}, One: {one_count}, Zero: {zero_count}, Others: {others}, Total Class = {total}")


def Preprocessing(xtrain, ytrain, xtest, ytest):
    # import seaborn as sns 

    # SCALLING USING Z-SCORE NORMALIZATION
    xtrain, xtest = Standard(xtrain, xtest)
    print("\nClass Frequency: Original Test Set")
    ClassCount(ytrain)
    # sns.countplot(x= ytrain)

    #REMOVING OUTLIERS USING LOCAL OUTLIER FACTOR
    xtrain, ytrain = LOF(xtrain, ytrain)
    print("\nClass Frequency: After Removing Outliers")
    ClassCount(ytrain)
    # sns.countplot(x= ytrain)

    # OVERSAMPLING USING SMOTE
    xtrain, ytrain = OverSample (xtrain, ytrain)
    print("\nClass Frequency: After Balancing")
    ClassCount(ytrain)
    print('\n')
    # sns.countplot(x= ytrain)

    return xtrain, ytrain, xtest, ytest


#Cook Models
def CookModels(x_train, ytrain):
    #Importing Classification Models
    from sklearn.naive_bayes import MultinomialNB,GaussianNB
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from xgboost.sklearn import XGBClassifier

    '''                             MODEL BUILDING                             '''
    SVCmodel = SVC().fit(x_train, ytrain)
    # MNBmodel = MultinomialNB().fit(x_train, ytrain)
    # GNBmodel = GaussianNB().fit(x_train, ytrain)
    DTmodel = DecisionTreeClassifier().fit(x_train, ytrain)
    RFmodel = RandomForestClassifier(n_estimators=1000).fit(x_train, ytrain)
    LRmodel = LogisticRegression().fit(x_train, ytrain)
    KNNmodel = KNeighborsClassifier(n_neighbors = 25).fit(x_train, ytrain)
    GBmodel  = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.01).fit(x_train, ytrain)
    AdaBmodel = AdaBoostClassifier(n_estimators=50).fit(x_train, ytrain)
    XGBmodel = XGBClassifier(n_estimators=1000).fit(x_train, ytrain)

    models = [SVCmodel,DTmodel, RFmodel, LRmodel, KNNmodel, GBmodel, AdaBmodel, XGBmodel]

    return models


#Evaluation
def Evaluation(xtest, ytest, models):
    from sklearn.metrics import classification_report as report
    for model in models:
        print(f"Result of: {model}")
        pred = model.predict(xtest)
        reports = report(ytest, pred)
        print(reports)
        print("\n")
