

path = "/content/mushrooms.csv"

import pandas as pd
df = pd.read_csv(path)
df.head()

df.shape

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

le = LabelEncoder()
df_encoded = df.apply(le.fit_transform, axis=0)
df_encoded.head()

df =df_encoded

X = df.iloc[:, 1:]
Y= df.iloc[:,0]

X

Y

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""#Prior Probability"""

def prior_prob(Y_train, label):
    total_examples = Y_train.shape[0]
    class_examples = np.sum(Y_train == label)
    return class_examples / float(total_examples)

"""#Likely hood"""

def cond_prob(X_train, y_train, feature_col, feature_val, label):
    X_filtered = X_train[y_train == label]
    numerator = np.sum(X_filtered.iloc[:, feature_col] == feature_val)
    denominator = X_filtered.shape[0]
    return numerator / float(denominator) if numerator > 0 else 1e-6

"""#Posterior Probability"""

def predict(X_train, Y_train, X_test):
    n_features = X_train.shape[1]
    classes = np.unique(Y_train)
    posterior_probab = []
    for label in classes:
        likelihood = 1.0
        for fea in range(n_features):
            cond = cond_prob(X_train, Y_train, fea, X_test.iloc[fea], label)
            likelihood *= cond
        prior = prior_prob(Y_train, label)
        post = likelihood * prior
        posterior_probab.append(post)
    pred = np.argmax(posterior_probab)
    return pred

"""#Accuracy of the model

"""

def accuracy(X_train, Y_train, X_test, Y_test):
    pred = []
    for i in range(X_test.shape[0]):
        p = predict(X_train, Y_train, X_test.iloc[i])
        pred.append(p)
    Y_pred = np.array(pred)
    acc = np.sum(Y_pred == Y_test) / Y_pred.shape[0]
    return acc

acc = accuracy(X_train, Y_train, X_test, Y_test)
print(acc)
