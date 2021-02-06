import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier as DecisionT
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Veri setini import ediyoruz
df=pd.read_csv("heart.csv", sep=",");

# Hastalık olup olmama dağılım grafiği
# sns.countplot(x="target", data=df, palette="bwr")

# Eğitim kümesi oluşturma. Train and Test

index=np.arange(0,13)
girdi = df.iloc[:,df.index[index]].values
cikti = df.iloc[:,13].values

## Bazı modeller scaler edilmiş verilerle daha iyi çalıştığı için scaler ediyoruz.
scaler=StandardScaler(with_mean=False)
scaler.fit(girdi)
girdi=scaler.fit_transform(girdi)

# X_train, X_test, y_train, y_test =tts(girdi,cikti,test_size = 0.25, random_state= 0)

models=[KNN(n_neighbors=10),SVC(),MultinomialNB(),GaussianNB(),DecisionT(),DecisionT()]

score_list=[0,0,0,0,0,0] 

parcalayici=KFold(n_splits=5,shuffle=True)

for train_index, test_index in parcalayici.split(girdi):
    train_x=girdi[train_index,:]
    test_x=girdi[test_index,:]
    train_y=cikti[train_index]
    test_y=cikti[test_index]
    i=0
    for model in models:
        model.fit(train_x,train_y)
        sonuc=model.score(test_x,test_y)
        score_list[i]+=sonuc
        i+=1
        print(f"{model} skor:{sonuc}")
    print("****************")

score_list=[i/5 for i in score_list] 
    
##  5 fold sonucu ortalama sonuclar;
# KNeighborsClassifier= 0.8085245901639343
# SVC =          0.8416393442622951
# MultinomialNB= 0.7984153005464482
# GaussianNB = 0.8017486338797815
# DecisionT= 0.7526229508196721
# DecisionT = 0.8183060109289617

# 

X_train, X_test, y_train, y_test =tts(girdi,cikti,test_size = 0.25, random_state= 0)
from sklearn.model_selection import GridSearchCV

##        SVM için model tuning
parametler={"C":np.arange(1,10),"kernel": ['rbf', 'poly', 'sigmoid']}

grid = GridSearchCV(SVC(),parametler,n_jobs=-1)
grid.fit(X_train,y_train)
print(f"svm için en iyi parametler :{grid.best_params_}" )


##      KNN için model tuning
parametler={"n_neighbors":np.arange(5,11)}
grid = GridSearchCV(KNN(),parametler,n_jobs=-1)
grid.fit(X_train,y_train)
print(f"KNN için en iyi parametler :{grid.best_params_}" )



























 

