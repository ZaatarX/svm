import sklearn
from sklearn import datasets, svm, metrics
from sklearn.neighbors import KNeighborsClassifier

cancer = datasets.load_breast_cancer()

print(cancer.feature_names)
print(cancer.target_names)

feature_data = cancer.data
label_data = cancer.target

feature_train, feature_test, label_train, label_test = sklearn.model_selection.train_test_split(
    feature_data, label_data, test_size=0.1)

cls = ['malignant', 'benign']

clf = svm.SVC(kernel='linear', C=1.5)
clf.fit(feature_train, label_train)

label_predictions = clf.predict(feature_test)

accuracy = metrics.accuracy_score(label_test, label_predictions)

print(accuracy)
