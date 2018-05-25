from utils.dataset import Dataset
from utils.improved import Baseline
from utils.scorer import report_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import learning_curve





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



