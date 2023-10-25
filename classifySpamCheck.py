import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

def aucCV(features, labels):
    imputer = SimpleImputer(missing_values=-1, strategy='mean')
    estimator = GradientBoostingClassifier(n_estimators=7)
    classifier = AdaBoostClassifier(estimator=estimator, n_estimators=31, learning_rate=0.92, random_state=42)
    model = make_pipeline(imputer, classifier)
    scores = cross_val_score(model, features, labels, cv=10, scoring='roc_auc')
    return scores

def predictTest(trainFeatures, trainLabels, testFeatures):
    imputer = SimpleImputer(missing_values=-1, strategy='mean')
    estimator = GradientBoostingClassifier(n_estimators=7)
    classifier = AdaBoostClassifier(estimator=estimator, n_estimators=31, learning_rate=0.92, random_state=42)
    model = make_pipeline(imputer, classifier)
    model.fit(trainFeatures, trainLabels)
    predictedTestLabels = model.predict_proba(testFeatures)[:, 1]
    return predictedTestLabels

if __name__ == "__main__":

    np.random.seed(42)
    data1 = np.loadtxt("spamTrain1.csv", delimiter=',')
    data2 = np.loadtxt("spamTrain2.csv", delimiter=',')
    data = np.vstack((data1, data2))
    shuffleIndex = np.arange(np.shape(data)[0])
    np.random.shuffle(shuffleIndex)
    data = data[shuffleIndex, :]
    features = data[:, :-1]
    labels = data[:, -1]

    print("The 10-fold cross-validation mean AUC is", np.mean(aucCV(features, labels)))

    trainFeatures, testFeatures, trainLabels, testLabels = train_test_split(features, labels, test_size=0.5, random_state=42)
    predictedTestLabels = predictTest(trainFeatures, trainLabels, testFeatures)
    print("The AUC for test set is", roc_auc_score(testLabels, predictedTestLabels))

    sortingIndex = np.argsort(testLabels)
    testExamples = testLabels.size
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(testExamples), testLabels[sortingIndex], 'b.')
    plt.xlabel('Sorted example no.')
    plt.ylabel('Real label')
    plt.subplot(2, 1, 2)
    plt.plot(np.arange(testExamples), predictedTestLabels[sortingIndex], 'r.')
    plt.xlabel('Sorted example no.')
    plt.ylabel('Predicted label')