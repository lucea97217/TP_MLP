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
        if X[i,0]==X[i,1]==1:
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
        if (X[i,0]==1 and X[i,1]==0) or (X[i,0]==0 and X[i,1]==1):
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

clfO = MLPClassifier(hidden_layer_sizes=(),activation='identity',solver='lbfgs').fit(X_train2, y_train2)
accuracy_score(y_test2,clfO.predict(X_test2))



# %% question 4 
# a
X_train3, X_test3, y_train3, y_test3 = train_test_split(X, Y3,test_size=0.1)
clfXOR = MLPClassifier(hidden_layer_sizes=(),activation='identity',solver='lbfgs').fit(X_train3, y_train3)
accuracy_score(y_test3,clfXOR.predict(X_test3))
# %%
#b
X_train4, X_test4, y_train4, y_test4 = train_test_split(X, Y3,test_size=0.1)

clfXOR = MLPClassifier(hidden_layer_sizes=(4,2),activation='identity',solver='lbfgs').fit(X_train3, y_train3)
accuracy_score(y_test4,clfXOR.predict(X_test4))
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

n_samples = len(digits.data)
x_train, x_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.1)


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
param1=[472, 'logistic', 'lbfgs']

#MLPClassifier(hidden_layer_sizes=(472),activation=('logistic'),solver='lbfgs')

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

df=MLPClassifier(hidden_layer_sizes=(param1[0]),activation=(param1[1]),solver=(param1[2]))
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



###### ANNEXE ############


# %%
L = []
for k in range(10) :
    s = 0
    for i in range(len(digits.target)) :
        if digits.target[i] == k :
            s = s +1
    L.append(s)
# %%
inter = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]

plt.hist(digits.target, bins=inter, rwidth=0.8)  # Création de l'histogramme
plt.xlabel('digits')
plt.xticks(np.arange(0, 10))
plt.ylabel('Effectif total')
plt.title("Répartition des digits par labels")
plt.show()

# %%
#%%
def transfo_color(Y):
    A = []
    for i in range(len(Y)):
        if Y[i]==0:
            A+=["#FF3030"]
        if Y[i]==1:
            A+=["#3D9140"]
    return A

X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
]
)

Y = [
    transfo_color(YOR(X)),
    transfo_color(YAND(X)),
    transfo_color(YXOR(X))
    ]
    
Title = ["OR", "AND", "XOR"]
u = np.linspace(0,1,10)
lines = [[0.5-u],[1.5-u],[]]

for i in range(len(Y)):
    plt.xlim([-0.2, 1.2])
    plt.ylim([-0.2, 1.2])
    plt.xticks([0,1])
    plt.yticks([0,1])
    plt.axis('off')
    #plt.title(Title[i])
    plt.scatter(
        X[:,0],
        X[:,1],
        s= 500,
        c = Y[i],
        marker="o")
    for j in range(len(Y[i])):
        plt.text(X[j][0]-0.04,X[j][1]-0.15,"("+str(X[j][0])+","+str(X[j][1])+")")
    for l in range(len(lines[i])):
        plt.plot(u,lines[i][l],color="b")
    plt.show()

# %%
