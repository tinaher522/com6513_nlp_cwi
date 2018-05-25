from utils.dataset import Dataset
from utils.improved import Baseline
from utils.scorer import report_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve

import time

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
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
    plt.savefig("Spa_LC_SVM_300")

def execute_demo(language):
    data = Dataset(language)


    print("{}: {} training - {} test".format(language, len(data.trainset), len(data.testset)))

    # for sent in data.trainset:
    #    print(sent['target_word'])#sent['sentence'], sent['target_word'], sent['gold_label'])

    baseline = Baseline(language)

    baseline.train(data.trainset)

    predictions_dev = baseline.test(data.devset)

    predictions_test = baseline.test(data.testset)

    gold_labels_dev = [sent['gold_label'] for sent in data.devset]
    gold_labels_test = [sent['gold_label'] for sent in data.testset]

    print("DEV result:")
    report_score(gold_labels_dev, predictions_dev, detailed=True)

    print("TEST result:")
    report_score(gold_labels_test, predictions_test, detailed=True)


if __name__ == '__main__':
    start = time.time()
    execute_demo('english')
    execute_demo('spanish')


    end = time.time()
    print("The running time is :",end - start)

