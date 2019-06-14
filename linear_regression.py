from pandas import read_csv
from sklearn import tree, model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2, norm
from sklearn.metrics import log_loss
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc
import math

import matplotlib.pyplot as plt
from itertools import cycle

seed = 112
test_size = 0.3
splits = 10

data = read_csv("basistabel-data-final.csv")

isYear = data['year']==2017 #Add this to filtered data, if only data from 2017 is wanted
isNotMissingData = data['missingData']==0

filteredData = data[isNotMissingData]

predictors = ['unemployment', 'immigrants', 'convicted', 'education', 'income']
X = filteredData[predictors].astype('float').to_numpy()
y = filteredData[['ghetto']].astype('float').to_numpy().flatten()

####    DECISION TREE ROC
dtc = tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced")

dt_tpr = np.array([])
dt_fpr = np.array([])
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed + i)

    dtc.fit(X_train, y_train)
    pred = dtc.predict(X_test)
    FP = len(np.where((pred == 1) & (pred != y_test))[0])   # False Positive
    FN = len(np.where((pred == 0) & (pred != y_test))[0])   # False Negative
    TN = len(np.where((pred == 0) & (pred == y_test))[0])   # True Negative
    TP = len(np.where((pred == 1) & (pred == y_test))[0])   # True Positive

    # Calculate rates
    tpr = TP / (TP + FN)
    fpr = FP / (FP + TN)

    # Add rates to array
    dt_tpr = np.append(dt_tpr, tpr)
    dt_fpr = np.append(dt_fpr, fpr)
    

####    LOGISTIC REGRESSION ROC
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

clf = LogisticRegression(random_state=seed, solver='lbfgs', class_weight="balanced").fit(X_train, y_train)
# y_score = clf.decision_function(X_test)
y_score = clf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)


####    PLOTTING ROC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.scatter(dt_fpr, dt_tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
# This is triggered by the criteria plot further down
#plt.show() 


####    K-FOLD CROSS VALIDATION
def kFoldTest(clf):
    kfold = model_selection.KFold(n_splits=splits, random_state=seed)

    scoring = 'accuracy'
    results = model_selection.cross_val_score(clf, X, y, cv=kfold, scoring=scoring)
    print('Accuracy: %.2f%% (+/- %.2f)' % (results.mean()*100, results.std()))

print('Testing the accuracy of Decision tree with entropy')
clf = tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced").fit(X_train, y_train)
kFoldTest(clf)

print('Testing the accuracy of Logistic Regression - lbfgs')
clf = LogisticRegression(random_state=seed, solver='lbfgs', class_weight="balanced")
kFoldTest(clf)


####    DECISION TREES

class Scores:
    highestScore = 0
    lowestScore = 1
    totalScore = 0
    averageScore = 1
    iterations = 0

def calculateAverageImportance(X, y, iterations=100):
    scores = Scores()

    bestTree = None

    importanceAvg = np.zeros((5))
    importances = np.zeros((iterations, 5))
    
    for i in range(iterations):
        dtc, score = createDecisionTree(X, y, i)

        importances[i] = dtc.feature_importances_
        importanceAvg += dtc.feature_importances_
        scores.totalScore += score

        if score > scores.highestScore:
            scores.highestScore = score
            bestTree = dtc

        if score < scores.lowestScore:
            scores.lowestScore = score

    scores.averageScore = scores.totalScore / iterations
    scores.iterations = iterations

    importanceAvg = importanceAvg / iterations

    return importances, importanceAvg, scores, bestTree
        

def createDecisionTree(X, y, i):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed + i)
    dtc = tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced")

    dtc.fit(X_train, y_train)

    score = dtc.score(X_test, y_test)

    return dtc, score

def printGraph(dtc):
     # Used for visualization
    dotData = tree.export_graphviz(dtc, out_file=None)
    print(dotData)

""" data = read_csv("basistabel-data-final.csv")

isYear = data['year']==2017
isNotMissingData = data['missingData']==0

filteredData = data[isYear & isNotMissingData]

predictors = ['unemployment', 'immigrants', 'convicted', 'education', 'income']
X = filteredData[predictors]
y = filteredData[['ghetto']] """

importances, importanceAvg, scores, bestTree = calculateAverageImportance(X, y, 10)

print('\nLooking at the importance of the criteria of decision trees')
print('Features:', predictors)
print('Average importance:', importanceAvg)

print('Highest score:', scores.highestScore)
print('Lowest score:', scores.lowestScore)
print('Average score:', scores.averageScore)

print('Creating a picture of the best performing tree: figures/bestDecisionTree.png')
tree.export_graphviz(bestTree, out_file='figures/bestDecisionTree.dot', 
                feature_names = predictors,
                class_names = ['non-ghetto', 'ghetto'],
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png using system command (requires Graphviz)
from subprocess import call
call(['dot', '-Tpng', 'figures/bestDecisionTree.dot', '-o', 'figures/bestDecisionTree.png', '-Gdpi=600'])

plt.figure()
lw = 2
for i in importances:
    plt.plot(np.arange(5), i, lw=1)
plt.plot(np.arange(5), importanceAvg, color='black', lw=lw, linestyle='--')
plt.ylim([0.0, (math.ceil(np.max(importances) * 10))/10 + 0.05])
plt.xlabel('Criteria')
plt.ylabel('Importance')
plt.xticks(np.arange(5), predictors)
plt.grid(axis='y')
plt.title('Criteria importance')
plt.show()