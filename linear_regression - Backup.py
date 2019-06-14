from pandas import read_csv
from sklearn import tree, model_selection
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2, norm
from sklearn.metrics import log_loss
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_curve, auc

import matplotlib.pyplot as plt
from itertools import cycle

def likelihood_ratio_test(features_alternate, labels, lr_model, features_null=None):
    """
    Compute the likelihood ratio test for a model trained on the set of features in
    `features_alternate` vs a null model.  If `features_null` is not defined, then
    the null model simply uses the intercept (class probabilities).  Note that
    `features_null` must be a subset of `features_alternative` -- it can not contain
    features that are not in `features_alternate`.
    Returns the p-value, which can be used to accept or reject the null hypothesis.
    """
    labels = np.array(labels)
    features_alternate = np.array(features_alternate)
    
    if features_null is not None:
        features_null = np.array(features_null)
        
        if features_null.shape[1] >= features_alternate.shape[1]:
            raise ValueError("Alternate features must have more features than null features")
        
        lr_model.fit(features_null, labels)
        null_prob = lr_model.predict_proba(features_null)[:, 1]
        df = features_alternate.shape[1] - features_null.shape[1]
    else:
        null_prob = sum(labels) / float(labels.shape[0]) * \
                    np.ones(labels.shape)
        df = features_alternate.shape[1]
    
    lr_model.fit(features_alternate, labels)
    alt_prob = lr_model.predict_proba(features_alternate)

    alt_log_likelihood = -log_loss(labels,
                                   alt_prob,
                                   normalize=False)
    null_log_likelihood = -log_loss(labels,
                                    null_prob,
                                    normalize=False)

    G = 2 * (alt_log_likelihood - null_log_likelihood)
    p_value = chi2.sf(G, df)

    return p_value

def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p

data = read_csv("basistabel-data-final.csv")

isYear = data['year']==2017
isNotMissingData = data['missingData']==0

filteredData = data[isYear & isNotMissingData]

predictors = ['unemployment', 'immigrants', 'convicted', 'education', 'income']
X = filteredData[predictors].astype('float').to_numpy()
y = filteredData[['ghetto']].astype('float').to_numpy().flatten()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

def testMethod(clf):
    #clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', class_weight="balanced").fit(X_train, y_train)
    print('Prediction for first:', clf.predict([X[0]]))
    print('Prediction probability for first:', clf.predict_proba([X[0]]))
    print('Score for test set:', clf.score(X_test, y_test))

    print('These values represents the propability that removing a criterion will yield a model, that is just as good')
    print('LR test (null function):\t', likelihood_ratio_test(X_train, y_train, clf))
    print('LR test (w/o unemployment):\t', likelihood_ratio_test(X_train, y_train, clf, np.delete(X_train[:, :], (0), axis=1)))
    print('LR test (w/o immigrants):\t', likelihood_ratio_test(X_train, y_train, clf, np.delete(X_train[:, :], (1), axis=1)))
    print('LR test (w/o convicted):\t', likelihood_ratio_test(X_train, y_train, clf, np.delete(X_train[:, :], (2), axis=1)))
    print('LR test (w/o education):\t', likelihood_ratio_test(X_train, y_train, clf, np.delete(X_train[:, :], (3), axis=1)))
    print('LR test (w/o income):\t', likelihood_ratio_test(X_train, y_train, clf, np.delete(X_train[:, :], (4), axis=1)))

    print('Logit:', logit_pvalue(clf, X_train))

    #sm_model = sm.Logit(y_train, sm.add_constant(X_train)).fit(disp=0)
    #print(sm_model.pvalues)
    #sm_model.summary()

""" clf = LogisticRegression(solver='liblinear', class_weight="balanced").fit(X_train, y_train)
print('\nTesting Liblinear')
testMethod(clf)
clf = LogisticRegression(random_state=0, solver='lbfgs', class_weight="balanced").fit(X_train, y_train)
print('\nTesting lbfgs')
testMethod(clf)
clf = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial', class_weight="balanced").fit(X_train, y_train)
print('\nTesting original lbfgs with multinomial')
testMethod(clf) """

clf = LogisticRegression(random_state=0, solver='lbfgs', class_weight="balanced").fit(X_train, y_train)
y_score = clf.decision_function(X_test)

decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced")
dTree = decisionTreeClassifier.fit(X_train, y_train)

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.plot(0.8, 0.8, '-ro')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

""" fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(1):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i]) """

print('Testing the accuracy of Decision tree')
seed = 2
splits = 10
clf = tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced").fit(X_train, y_train)
kfold = model_selection.KFold(n_splits=splits, random_state=seed)

scoring = 'accuracy'
results = model_selection.cross_val_score(clf, X, y, cv=kfold, scoring=scoring)
print('Accuracy -val set: %.2f%% (+/- %.2f)' % (results.mean()*100, results.std()))

print('Testing the accuracy of Logistic Regression')
clf = LogisticRegression(random_state=seed, solver='lbfgs', class_weight="balanced")

scoring = 'accuracy'
results = model_selection.cross_val_score(clf, X, y, cv=kfold, scoring=scoring)
print('Accuracy -val set: %.2f%% (+/- %.2f)' % (results.mean()*100, results.std()))
"""
#split data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
#fit model
clf.fit(X_train, y_train)
#accuracy on test set
result = clf.score(X_test, y_test)
print("Accuracy - test set: %.2f%%" % (result*100.0))

print('\nTesting the logloss')
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=test_size, random_state=seed)
model.fit(X_train, y_train)
#predict and compute logloss
pred = model.predict(X_test)
accuracy = log_loss(y_test, pred)
print("Logloss: %.2f" % (accuracy)) """

exit()

class Scores:
    highestScore = 0
    lowestScore = 1
    totalScore = 0
    averageScore = 1
    iterations = 0

def calculateAverageImportance(X, y, iterations=100):
    scores = Scores()

    bestTree = None

    importance = [5]
    
    for _ in range(iterations):
        dTree, score = createDecisionTree(X, y)

        importance += dTree.feature_importances_
        scores.totalScore += score

        if score > scores.highestScore:
            scores.highestScore = score
            bestTree = dTree

        if score < scores.lowestScore:
            scores.lowestScore = score

    scores.averageScore = scores.totalScore / iterations
    scores.iterations = iterations

    importance /= iterations

    return importance, scores, bestTree
        

def createDecisionTree(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

    decisionTreeClassifier = tree.DecisionTreeClassifier(criterion="entropy", class_weight="balanced")
    dTree = decisionTreeClassifier.fit(X_train, y_train)

    score = dTree.score(X_test, y_test)

    return dTree, score

def printGraph(dTree):
     # Used for visualization
    dotData = tree.export_graphviz(dTree, out_file=None)
    print(dotData)

data = read_csv("basistabel-data-final.csv")

isYear = data['year']==2017
isNotMissingData = data['missingData']==0

filteredData = data[isYear & isNotMissingData]

predictors = ['unemployment', 'immigrants', 'convicted', 'education', 'income']
X = filteredData[predictors]
y = filteredData[['ghetto']]

importance, scores, bestTree = calculateAverageImportance(X, y)

print('Features:', predictors)
print('Importance:', importance)

print('Highest score:', scores.highestScore)
print('Lowest score:', scores.lowestScore)
print('Average score:', scores.averageScore)

print('The tree of the most best scoring:')
printGraph(bestTree)