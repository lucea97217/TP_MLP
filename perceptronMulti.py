'''

PERCEPTRON MULTICOUCHE  - MLP

'''
#%%
import numpy.random as rd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split , GridSearchCV
import numpy as np
from sklearn.metrics import accuracy_score
import random as r
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn import tree, datasets
from sklearn.svm import SVC
from svm_source import *
from sklearn import svm
#%%

def randX(n):
    x=rd.randint(0,2,(n,2))
    return(x)


#%%
def YAND(X):
    C = []
    for i in range(0,X.shape[0]):
        if X[i,0]==X[i,1]:
            C.append(1)
        else:
            C.append(0)
    return(C)



#%%
def YOR(X):
    D = []
    for i in range(0,X.shape[0]):
        if X[i,0]==1 or X[i,1]==1:
            D.append(1)
        else:
            D.append(0)
    return(D)
#%%
def YXOR(X):
    E = []
    for i in range(0,X.shape[0]):
        if (X[i,0]==1 and X[i,1]!=1) or (X[i,0]!=1 and X[i,1]==1):
            E.append(1)
        else:
            E.append(0)
    return(E)

# %%

X = randX(100)
Y1 = YAND(X)
Y2 = YOR(X)
Y3 = YXOR(X)

# %% question 2

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, Y1,test_size=0.1)
clfAND = MLPClassifier(hidden_layer_sizes=(),activation='identity',solver='lbfgs').fit(X_train1, y_train1)


print("score : ",accuracy_score(y_test1,clfAND.predict(X_test1)))

#%% question 3


X_train2, X_test2, y_train2, y_test2 = train_test_split(X, Y2,test_size=0.1)

clfOR = MLPClassifier(hidden_layer_sizes=(),activation='identity',solver='lbfgs').fit(X_train2, y_train2)
accuracy_score(clfOR)



# %% question 4 
# a
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y3,test_size=0.1)
clfXOR = MLPClassifier(hidden_layer_sizes=(),activation='identity',solver='lbfgs').fit(X_train3, y_train3)
accuracy_score(clfXOR)
# %%
#b
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, Y3,test_size=0.1)

clfXOR = MLPClassifier(hidden_layer_sizes=(4,2),activation='identity',solver='lbfgs').fit(X_train3, y_train3)
accuracy_score(clfXOR)
# %%
#c

X_train5, X_test5, y_train5, y_test5 = train_test_split(X, Y3,test_size=0.1)
score4 = np.zeros(100)
for j in range (0,len(score4)):
    clfXOR = MLPClassifier(hidden_layer_sizes=(4,2),activation='tanh',solver='lbfgs').fit(X_train5, y_train5)
    score4[j]=clfXOR.score(X_test5, y_test5)
    
print("score ",score4) # 
# %%
digits = load_digits()
#spliting the dataset in train and test
n_samples = len(digits.images)
#digits.images = data = digits.images.reshape((n_samples, -1))
x_train, x_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.1)
#preprocessing data
#scaler = StandardScaler()
#scaler.fit(x_train) 

#%% couche 
test = 100
score = 0
param = [0,'','','']
for parameter in range(test):
    
    hls = r.randint(100, 1000)
    activ = r.choice(['identity', 'logistic', 'tanh', 'relu'])
    lr = r.choice(['constant', 'invscaling', 'adaptive'])
    solv = r.choice(['lbfgs', 'sgd', 'adam'])
    clfdig= MLPClassifier(hidden_layer_sizes=hls,activation=activ,learning_rate=lr,solver=solv).fit(x_train,Y_train)
    scoreclf = accuracy_score(Y_test,clfdig.predict(x_test))
    if scoreclf > score:
        param[0]=hls
        param[1]=activ
        param[2]=lr
        param[3]=solv
        score = scoreclf
#%%
param1=[472, 'logistic', 'invscaling', 'lbfgs']

# %%

test = 10
score2 = score
param2 = [0,'','','']
score3=0
for parameter in range(test):
    
    hls = r.randint(100, 1000)
    activ = r.choice(['identity', 'logistic', 'tanh', 'relu'])
    lr = r.choice(['constant', 'invscaling', 'adaptive'])
    solv = r.choice(['lbfgs', 'sgd', 'adam'])
    clfdig1= MLPClassifier(hidden_layer_sizes=(param1[0],hls),activation=(param1[1]),learning_rate=(param1[2]),solver=(param1[3])).fit(x_train,Y_train)
    scoreclf = accuracy_score(Y_test,clfdig1.predict(x_test))
    if scoreclf > score2:
        param[0]=hls
        param[1]=activ
        param[2]=lr
        param[3]=solv
        score3 = scoreclf
        
#Useless 2 couches



#%%

df=MLPClassifier(hidden_layer_sizes=(param[0]),activation=(param1[1]),learning_rate=(param1[2]),solver=(param1[3]))
acc = cross_val_score(df, x_train, Y_train, cv=5, n_jobs=-1)
print("The error using MLP method equals {:.3f}%.".format((1 - acc.mean()) * 100))
#%%

#linear classifier SVM

parameters = {'kernel': ['linear'], 'C': list(np.logspace(-3, 3, 20))}
svr = svm.SVC()
acc = cross_val_score(svr, x_train, Y_train, cv=5)
clf_linear = GridSearchCV(svr, parameters)
clf_linear.fit(x_train, Y_train)
print('Generalization score for linear kernel: %s' %((1-max(sorted(clf_linear.cv_results_['mean_test_score'])))*100),'%')

# %%
# fit a classifier (poly) and test all the Cs
parameters = {'kernel': ['poly'], 'C': list(np.logspace(-3, 3, 20))}
svr = svm.SVC()
clf_linear = GridSearchCV(svr, parameters)
clf_linear.fit(x_train, Y_train)

print('Generalization score for linear kernel: %s' %((1-max(sorted(clf_linear.cv_results_['mean_test_score'])))*100),'%')
# %%
# fit a classifier (rbf) and test all the Cs
parameters = {'kernel': ['rbf'], 'C': list(np.logspace(-3, 3, 20))}
svr = svm.SVC()
clf_linear = GridSearchCV(svr, parameters)
clf_linear.fit(x_train, Y_train)

print('Generalization score for linear kernel: %s' %((1-max(sorted(clf_linear.cv_results_['mean_test_score'])))*100),'%')
# %%
# fit a classifier (sigmoid) and test all the Cs
parameters = {'kernel': ['sigmoid'], 'C': list(np.logspace(-3, 3, 20))}
svr = svm.SVC()
clf_linear = GridSearchCV(svr, parameters)
clf_linear.fit(x_train, Y_train)

print('Generalization score for linear kernel: %s' %((1-max(sorted(clf_linear.cv_results_['mean_test_score'])))*100),'%')
# %%