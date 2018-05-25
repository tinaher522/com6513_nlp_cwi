from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RepeatedKFold
import nltk
from numpy import array
import numpy as np
from sklearn import preprocessing

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

'''
this program select DecisionTreeClassifier implemented with spanish model and the VotingClassifier (soft voting) with english model
the process of adjusting parameter of different model and comebine VotingClassifier shows in model_selection.py
'''

def plot_learning_curve(estimator, title, X, y, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

class Baseline(object):
    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            clf1 = LogisticRegression(random_state=1)
            clf2 = RandomForestClassifier(random_state=1)
            clf3 = GaussianNB()
            self.model = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='soft',
                                     weights=[3, 5, 2])
        else:  # spanish
            self.avg_word_length = 6.2
            self.model = DecisionTreeClassifier(max_depth=None, min_samples_split=5, random_state=0)

    def word_frequency(self,trainset):
        word = []
        for sent in trainset:
            raw_sent = sent['sentence'].split()
            for i in raw_sent:
                word.append(i)
        word_frequency = nltk.FreqDist(word)
        return word_frequency

    def pos_list(self, trainset):
        pos_list = []
        for sent in trainset:
            tagged = nltk.pos_tag(sent['target_word'])[0][1]
            pos_list.append(tagged)
        pos_list = list(set(pos_list))
        pos_list = array(pos_list)

        return pos_list



    def extract_features(self, word):
        len_tokens = len(word.split(' '))
        len_chars = len(word) / self.avg_word_length

        return [len_tokens,len_chars]


    def train(self,trainset):
        X = []
        y = []
        wf = self.word_frequency(trainset)
        #pos=self.pos_list(trainset)
        for sent in trainset:
            x=self.extract_features(sent['target_word'])
            x.append(wf[sent['target_word']])
            #x.append(pos[sent['target_word']])
            X.append(x)
            #X = preprocessing.scale(X)
            y.append(sent['gold_label'])
        if self.language == 'english':
            title="Learning Curves (VotingClassifier (soft voting))"
            self.model.fit(X, y)
            rkf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=0)
            plot_learning_curve(self.model, title, X, y, cv=rkf, n_jobs=1)
        else:
            title="Learning Curves (DecisionTreeClassifier)"
            self.model.fit(X, y)
            rkf = RepeatedKFold(n_splits=10, n_repeats=2, random_state=0)
            plot_learning_curve(self.model, title, X, y, cv=rkf, n_jobs=1)
        plt.show()


    def test(self, testset):
        X = []
        wf= self.word_frequency(testset)
        #pos = self.pos_list(testset)
        for sent in testset:
            x= self.extract_features(sent['target_word'])
            x.append(wf[sent['target_word']])
            #x.append(pos[sent['target_word']])
            X.append(x)
            #X=preprocessing.scale(X)

        return self.model.predict(X)


