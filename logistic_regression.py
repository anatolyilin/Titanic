import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('../train.csv')
dataset_test = pd.read_csv('../test.csv')

# analyze the data
dataset_train.info()
# null in Age, Cabin and Embarked

# Age -> determine the spread.
plt.title('Scatter of Age')
plt.xlabel('Passenger')
plt.ylabel('Age')
plt.scatter(dataset_train["PassengerId"], dataset_train["Age"])
plt.plot(dataset_train["PassengerId"], dataset_train["Age"].median()*np.ones([len(dataset_train["PassengerId"]),1]),c ="red")
plt.show()
"""
dataset_train["Age"].max()
dataset_train["Age"].min()
"""
#oldest is 80, youngest 0.42 ~ 0.
# mean is arround 30, while histogram (bin size 8 -> 0-80, so 10y per bin) shows it's between 20 and 30.
plt.hist(dataset_train["Age"].dropna(), bins=8)
plt.show()

#since the oldest is 80, lets set 80 bins:
plt.hist(dataset_train["Age"].dropna(), bins=80)

# anywhere between 20-30 would seem like a good average
from scipy.stats import norm
mean = dataset_train["Age"].dropna().mean()
std = dataset_train["Age"].dropna().std()
x = np.linspace(0,100,10000)
y = norm.pdf(x, loc=mean, scale=std)    # add standard deviation of the age
plt.plot(x,y*1000)
plt.show()

# let's plot survival rate per age group
plt.hist(dataset_train["Age"].dropna(), bins=80, histtype="step", color="red")
plt.hist(dataset_train["Survived"], bins=80, histtype="step", color="blue")
plt.show()


def preprocess(dataset):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    le2 = preprocessing.LabelEncoder()
    dataset['Embarked'] = le2.fit_transform(dataset['Embarked'].astype(str) )
    # remove NaN in age coloumn
    imputer = preprocessing.Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
    imputer = imputer.fit(dataset['Age'].reshape(-1, 1))
    dataset['Age'] = imputer.transform(dataset['Age'].reshape(-1, 1))
    imputer3 = preprocessing.Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = 0)
    imputer3 = imputer3.fit(dataset['Fare'].reshape(-1, 1))
    dataset['Fare'] = imputer3.transform(dataset['Fare'].reshape(-1, 1))
    return dataset

#dataset_train = preprocess(dataset_train)
#dataset_test = preprocess(dataset_test)