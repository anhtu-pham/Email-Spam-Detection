import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve
from classifySpam import predictTest

desiredFPR = 0.01

def tprAtFPR(labels,outputs,desiredFPR):
    fpr, tpr, _ = roc_curve(labels,outputs)
    # True positive rate for highest false positive rate < 0.01
    maxFprIndex = np.where(fpr<=desiredFPR)[0][-1]
    fprBelow = fpr[maxFprIndex]
    fprAbove = fpr[maxFprIndex + 1]
    # Find TPR at desired FPR by linear interpolation
    tprBelow = tpr[maxFprIndex]
    tprAbove = tpr[maxFprIndex + 1]
    tprAt = (tprAbove - tprBelow) / (fprAbove-fprBelow) * (desiredFPR - fprBelow) + tprBelow
    return tprAt, fpr, tpr

data1 = np.loadtxt('data1.csv', delimiter=',')
data2 = np.loadtxt('data2.csv', delimiter=',')
trainData = np.vstack((data1, data2))
testData = np.loadtxt('testData.csv', delimiter=',')

# Randomly shuffle rows of training and test sets, then separate labels
shuffleIndex = np.arange(np.shape(trainData)[0])
np.random.shuffle(shuffleIndex)
trainData = trainData[shuffleIndex,:]
trainFeatures = trainData[:, :-1]
trainLabels = trainData[:, -1]

shuffleIndex = np.arange(np.shape(testData)[0])
np.random.shuffle(shuffleIndex)
testData = testData[shuffleIndex, :]
testFeatures = testData[:, :-1]
testLabels = testData[:, -1]

testOutputs = predictTest(trainFeatures, trainLabels, testFeatures)
aucTestRun = roc_auc_score(testLabels, testOutputs)
tprAtDesiredFPR, fpr, tpr = tprAtFPR(testLabels, testOutputs, desiredFPR)

plt.plot(fpr, tpr)

print("Test set AUC is", aucTestRun)
print("TPR at FPR =", desiredFPR, "is", tprAtDesiredFPR)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curve for email spam detector")    
plt.show()