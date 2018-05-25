from utils.dataset import Dataset
from utils.improved import Baseline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_blobs
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

##this script is used to adjust models' parameters and select suitable model for the project

def extract_features(word):

    len_chars = len(word) / 6.2
    len_tokens = len(word.split(' '))

    return [len_chars, len_tokens]

#data = Dataset("English")
data = Dataset("Spanish")
X = []
y = []
for sent in data.trainset:
    X.append(extract_features(sent['target_word']))
    y.append(sent['gold_label'])

#Weighted Average Probabilities (Soft Voting)
##Using the VotingClassifier with GridSearch
clf1=LogisticRegression(random_state=1)
clf2=RandomForestClassifier(random_state=1)
clf3=GaussianNB()
eclf1=VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft',weights=[3,5,2])
params={'lr__C': [1.0, 100.0], 'rf__n_estimators': [20, 200],}
grid=GridSearchCV(estimator=eclf1, param_grid=params, cv=5)
grid=grid.fit(X,y)
print(grid)
for clf, label in zip([clf1, clf2, clf3, eclf1], ['Logistic Regression', 'Random Forest', 'GaussianNB' ,'Ensemble1']):
    scores=cross_val_score(clf,X,y,cv=5,scoring="accuracy")
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))


clf4=DecisionTreeClassifier(max_depth=None,min_samples_split=5,random_state=0)
clf5=KNeighborsClassifier(n_neighbors=7)
clf6=SVC()
eclf2=VotingClassifier(estimators=[('dt', clf4), ('kn', clf5), ('svc', clf6)], voting='soft',weights=[3,5,2])
for clf, label in zip([clf4, clf5, clf6, eclf2], ['Decision Tree', 'KNeighbors', 'SVC' ,'Ensemble2']):
    scores=cross_val_score(clf,X,y,cv=5,scoring="accuracy")
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

def VotingClassifier():
    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = GaussianNB()
    eclf1 = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft', weights=[3, 5, 2])









#----------------compare different sklearn models' perforamce and adjust the parameter------------#

###model selection l1,0.2 0.720173839117

#1.LogisticRegression
#logreg = LogisticRegression(penalty='l1', tol=0.2)
#print (cross_val_score(logreg, X, y, cv=10, scoring='accuracy').mean())



###ensemble methods

##2.BaggingClassifier (0.5,0.5,0.713285632928)
#bagging=BaggingClassifier(KNeighborsClassifier(),max_samples=0.5,max_features=0.5)
#print (cross_val_score(bagging, X, y, cv=10, scoring='accuracy').mean())


###3.RandomForestClassifier (3,0.73478839312)(3,....,0.73453186207)稳定
#clf = RandomForestClassifier(n_estimators=3,max_depth=None,min_samples_split=5,random_state=0)
#print (cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean())

####4.DecisionTreeClassifier 0.734238915686
#clf = DecisionTreeClassifier(max_depth=None,min_samples_split=5,random_state=0)
#print (cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean())

####5.ExtraTreesClassifier 0.734422025641
#clf=ExtraTreesClassifier(n_estimators=10,max_depth=None,min_samples_split=5,random_state=0)
#print (cross_val_score(clf, X, y, cv=10, scoring='accuracy').mean())


######6.AdaBoostClassifier (100,0.730796025969)
#clf=AdaBoostClassifier(n_estimators=100)
#scores=cross_val_score(clf,X,y)
#print(scores.mean())


#######7.GradientBoostingClassifier 0.734971976995
#clf=GradientBoostingClassifier(n_estimators=100,learning_rate=1.0,max_depth=1, random_state=0).fit(X, y)
#print(clf.score(X,y))


#VotingClassifier
########8.Majority Class Labels (Majority/Hard Voting)

#clf1=LogisticRegression(random_state=1)
#clf2=RandomForestClassifier(random_state=1)
#clf3=GaussianNB()
#clf4=AdaBoostClassifier(n_estimators=100)
#eclf=VotingClassifier(estimators=[("lr",clf1),("rf",clf2),("gnb",clf3),("ab",clf4)],voting="hard")
#for clf, label in zip([clf1, clf2, clf3, clf4, eclf], ['Logistic Regression', 'Random Forest', 'naive Bayes', 'AdaBoost','Ensemble']):
    #scores=cross_val_score(clf,X,y,cv=5,scoring="accuracy")
    #print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))












