from sklearn.svm import SVC
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

# data loading 
data_train = "../classification_dataset/classification_train.csv"
df_data_train = pd.read_csv(data_train)
X = np.array(df_data_train.iloc[:,0:2])
y = np.array(df_data_train.iloc[:,-1])

# data splitting
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.2, random_state=1)

# Feature Scaling
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Training a SVM classifier
svm = SVC(kernel= 'rbf', random_state=1, C=0.1)
svm.fit(X_train, y_train)
 
# Mode performance
y_pred = svm.predict(X_test)
print('Accuracy: %.3f' %accuracy_score(y_test, y_pred))

# prediction on test data
data_test = "../classification_dataset/classification_test.csv"
data = pd.read_csv(data_test)
X = np.array(data[['x_1','x_2']])
Y = svm.predict(X)
predicted_Y = pd.DataFrame(Y)
print('=============Saving prediction===================================="')
predicted_Y.to_csv('predicted_svm_sklearn.csv', index = False, header = False)
print('=============Saved prediction===================================="')