import pandas
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

print("DECISION TREE")

dataFrame1 = pandas.read_excel("data.xlsx")
feature1 = dataFrame1.drop("Gender", axis=1)
label1 = dataFrame1["Gender"] 
Classifier1 = DecisionTreeClassifier()
Classifier1.fit(feature1.values, label1)
height1 = input("Enter your height in feet: ")
weight1 = input("Enter your weight in kg: ")
footSize1 = input("Enter your foot size in inches: ")
prediction1 = Classifier1.predict([[height1, weight1, footSize1]])
print("Prediction that you're:", prediction1[0])

print("NAIVE BASE")

dataFrame2 = pandas.read_excel("data.xlsx")
feature2 = dataFrame2.drop("Gender", axis=1)
label2 = dataFrame2["Gender"] 
Classifier2 = GaussianNB()
Classifier2.fit(feature2.values, label2)
height2 = input("Enter your height in feet: ")
weight2 = input("Enter your weight in kg: ")
footSize2 = input("Enter your foot size in inches: ")
prediction2 = Classifier2.predict([[height2, weight2, footSize2]])
print("Prediction that you're:", prediction2[0])

print("KNN")

dataFrame3 = pandas.read_excel("data.xlsx")
feature3 = dataFrame3.drop("Gender", axis=1)
label3 = dataFrame3["Gender"] 
Classifier3 = KNeighborsClassifier(n_neighbors=3)
Classifier3.fit(feature3.values, label3)
height3 = input("Enter your height in feet: ")
weight3 = input("Enter your weight in kg: ")
footSize3 = input("Enter your foot size in inches: ")
prediction3 = Classifier3.predict([[height3, weight3, footSize3]])
print("Prediction that you're:", prediction3[0])


nb_samples = 1000
x, y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2,
                           n_redundant=0, n_clusters_per_class=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model1 = LogisticRegression()
model1.fit(xtrain, ytrain)
print("Accuracy of Decision Tree",accuracy_score(ytest, model1.predict(prediction1)))
model2 = LogisticRegression()
model2.fit(xtrain, ytrain)
print("Accuracy of Naive Base",accuracy_score(ytest, model2.predict(prediction2)))
model3 = LogisticRegression()
model3.fit(xtrain, ytrain)
print("Accuracy of KNN",accuracy_score(ytest, model3.predict(prediction3)))


'''from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
nb_samples = 1000
x, y = make_classification(n_samples=nb_samples, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(xtrain, ytrain)
print(accuracy_score(ytest, model.predict(xtest)))'''