import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.threshold = None
        self.feature_idx = None
        self.alpha = None
        
    def predict(self,X):
        n_samples = X.shape[0]
        X_c = X[:,self.feature_idx]
        preds = np.ones(n_samples)
        
        if self.polarity ==1:
            preds[X_c < self.threshold] = -1
        else:
            preds[X_c > self.threshold] = -1
            
        return preds

class myAdaBoost:
    def __init__(self,n_clf=5):
        self.n_clf = n_clf
        
    def fit(self,X,y):
        n_samples,n_features = X.shape
        w = np.full(n_samples, (1/n_samples))
        
        self.clfs=[]
        for _ in range(self.n_clf):
            adaBoost_classifier = DecisionStump()
            min_error = float('inf')
            for feat in range(n_features):
                X_c = X[:,feat]
                thresholds=np.unique(X_c)
                for threshold in thresholds:
                    p=1
                    preds=np.ones(n_samples)
                    preds[X_c<threshold]=-1
                    
                    misclassified = w[y!=preds]
                    error=sum(misclassified)
                    
                    if error >0.5:
                        p=-1
                        error=1-error
                    
                    if error<min_error:
                        min_error=error
                        adaBoost_classifier.threshold=threshold
                        adaBoost_classifier.feature_idx=feat
                        adaBoost_classifier.polarity=p
            
            EPS=1e-10
            adaBoost_classifier.alpha=0.5*np.log((1.0-min_error+EPS)/(min_error+EPS))
            preds = adaBoost_classifier.predict(X)
            w *= np.exp(-adaBoost_classifier.alpha*y*preds)
            w/=np.sum(w)
            self.clfs.append(adaBoost_classifier)
            
    def predict(self,X):
        clf_preds = [adaBoost_classifier.alpha*adaBoost_classifier.predict(X) for adaBoost_classifier in self.clfs]
        y_pred = np.sum(clf_preds,axis=0)
        y_pred = np.sign(y_pred)
        return y_pred

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    return accuracy
  
# loading train data
data_train = "../classification_dataset/classification_train.csv"
data_train = pd.read_csv(data_train)
X = data_train.iloc[:,0:2].values
y = data_train.iloc[:,-1].values

# y[y==0]=-1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=58)

adaBoost_classifier = myAdaBoost()
adaBoost_classifier.fit(X_train, y_train)
y_pred = adaBoost_classifier.predict(X_test)
acc = accuracy(y_test, y_pred)

print(f"The Accuracy of the AdaBoost classifier is {acc}")
print('============')

# Testing classifier on test set
data_train = "../classification_dataset/classification_test.csv"
data = pd.read_csv(data_train)
X = np.array(data[['x_1','x_2']])
Y = adaBoost_classifier.predict(X)
predicted_Y = pd.DataFrame(Y)
print('Saved prediction to "predicted_adaBoost.csv"')
predicted_Y.to_csv('predicted_adaBoost.csv', index = False, header = False)