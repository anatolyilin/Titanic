# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def encodeFem(dataset):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    dataset['Sex'] = le.fit_transform(dataset['Sex'])
    return dataset


def ageDeNAN(dataset, type_st, col_name):
    from sklearn import preprocessing
    # remove NaN in age coloumn
    imputer = preprocessing.Imputer(missing_values = 'NaN', strategy = type_st, axis = 0)
    imputer = imputer.fit(dataset[col_name].reshape(-1, 1))
    dataset[col_name] = imputer.transform(dataset[col_name].reshape(-1, 1))
    return dataset

# %%
dataset_train = pd.read_csv('../train.csv')
dataset_test = pd.read_csv('../test.csv')

# %%
# analyze the data
dataset_train.info()
# null in Age, Cabin and Embarked

# %%
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

# %%
# let's plot survival rate per age group
plt.hist(dataset_train["Age"].dropna(), bins=np.arange(0,80,2.5), histtype="step", color="blue",label="passanger distribution")
hs = dataset_train['Age'] *  dataset_train['Survived']
dataset_train = encodeFem(dataset_train)
plt.hist(hs[hs>0], bins=np.arange(0,80,2.5), histtype="step" , color="red", label="survived")
# male are encoded as 1, histogram of male survive rate per age group
hsf = dataset_train['Age'] *  dataset_train['Survived'] * dataset_train['Sex']
plt.hist(hsf[hsf>0], bins=np.arange(0,80,2.5), histtype="step" , color="black", label="survived and men")
plt.legend()
plt.show()

plt.hist(dataset_train["Age"].dropna(), bins=np.arange(0,80,2.5), histtype="step", color="blue",label="passanger distribution")
hs2 = dataset_train['Age']* dataset_train['Sex']
plt.hist(hs2[hs2>0], bins=np.arange(0,80,2.5), histtype="step" , color="red", label="male passangers")
plt.legend()
plt.show()

# concretely, this means len(hsf[hsf>0]) 93 survivors were men, 
# from len(dataset_train["Survived"][dataset_train["Survived"]>0]) 342 passangers
# so 27% of survivors were men.
# dataset_train["Survived"].mean() 38% of the passangers survived
# any female had 0.38*(1-0.27) = 0.2774 chance of survival
# so blindly, setting surval = 1 when female = 1, should give accuracy of roughly 27,74

# %%
#since the oldest is 80, lets set 80 bins:
plt.hist(dataset_train["Age"].dropna(), bins=80, label="age distribution before adjusting")

# anywhere between 20-30 would seem like a good average
from scipy.stats import norm
mean = dataset_train["Age"].dropna().mean()
std = dataset_train["Age"].dropna().std()
x = np.linspace(0,100,10000)
y = norm.pdf(x, loc=mean, scale=std)    # add standard deviation of the age
plt.plot(x,y*1000, label="normal distribution before adjusting")
plt.legend()


# time to address NaN in the age coloumn.
"""
dataset_train2 = ageDeNAN(dataset_train, "mean", "Age")

# replot histogram and normal distribution, it should look similar (shouldn't be skewed)
from scipy.stats import norm
mean = dataset_train2["Age"].dropna().mean()
std = dataset_train2["Age"].dropna().std()
x = np.linspace(0,100,10000)
y = norm.pdf(x, loc=mean, scale=std)    # add standard deviation of the age
plt.plot(x,y*1000, color = "green", label="normal distribution after adjusting [mean]")
"""
dataset_train3 = ageDeNAN(dataset_train, "median", "Age")

# replot histogram and normal distribution, it should look similar (shouldn't be skewed)
from scipy.stats import norm
mean = dataset_train3["Age"].dropna().mean()
std = dataset_train3["Age"].dropna().std()
x = np.linspace(0,100,10000)
y = norm.pdf(x, loc=mean, scale=std)    # add standard deviation of the age
plt.plot(x,y*1000, color = "red", label="normal distribution after adjusting [median]")
"""
dataset_train4 = ageDeNAN(dataset_train, "most_frequent", "Age")

# replot histogram and normal distribution, it should look similar (shouldn't be skewed)
from scipy.stats import norm
mean = dataset_train4["Age"].dropna().mean()
std = dataset_train4["Age"].dropna().std()
x = np.linspace(0,100,10000)
y = norm.pdf(x, loc=mean, scale=std)    # add standard deviation of the age
plt.plot(x,y*1000, color = "black", label="normal distribution after adjusting [most freq]")
"""
plt.legend()
plt.show()

# mean and median look somewhat similar on the graph, let's proceed with the median.
dataset_train = ageDeNAN(dataset_train, "median", "Age")
# %%
dataset_train.info()
# Cabin has a lot of missing data. Let's remove it
dataset_train = dataset_train.iloc[:, [0,1,2,3, 4,5,6,7,9,11]]
# %%
dataset_train.info()
# Embarked coloumn is incomplete. 
# We can fill it with most frequent values, leave NaN as a different label or ???
# Let's see if there's a direct 1-to-1 correlation between "Embarked" and another coloumn 
from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()

embarked_data = dataset_train.copy(deep=True)
embarked_data.dropna(axis=0, how="any" , inplace = True)
embarked_data['Embarked'] = le2.fit_transform(embarked_data['Embarked'].astype(str) )
embarked_data.info()
# %%

#%%
# no correlation (-0.163)
print(embarked_data['Embarked'].corr( embarked_data['Survived'], method="pearson"))
# -0.221
print(embarked_data['Embarked'].corr(embarked_data['Fare'], method="pearson"))
#-0.0142
print(embarked_data['Embarked'].corr(embarked_data['Age'], method="pearson"))
# 0.104
print(embarked_data['Embarked'].corr(embarked_data['Sex'], method="pearson"))
# 0.157
print(embarked_data['Embarked'].corr(embarked_data['Pclass'], method="pearson"))

# no clear 1-to-1 correletaion between embarking and any other metric. 
# weak correlation between Fare, Survived, PClass and Sex
# %%
# Plot histogram of Embarked, to see if there's a trend:
plt.hist(embarked_data['Embarked'], bins=[0,1,2,3], label="Ports historgram")
# Majority of passangers came from port 3 (le2.classes_ ) = "S"
# Fill in with the majority is ok.
dataset_train.fillna(value="S",inplace = True)
# Encode letters to digits
from sklearn import preprocessing
le2 = preprocessing.LabelEncoder()
dataset_train['Embarked'] = le2.fit_transform(dataset_train['Embarked'].astype(str) )

#%%
# Print all cols having 0. We don't expect any in Pclass, Age and Fare. But we do in Fare. 
print((dataset_train == 0).astype(int).sum(axis=0))

# Expecting a strong correlation between Pclass and Fare:
print(dataset_train['Fare'].corr(dataset_train['Pclass'], method="pearson"))

# about -0.55 Perhaps some values are incorrect? E.g. 0? 
# Is price of 0 indicator or is it error (lack of data)?

#%%
# Let's see the mean and std FARE per Pclass:
x = np.linspace(0,500,100000)

mean3 = dataset_train["Fare"][dataset_train["Pclass"] == 3 ].mean()
std3 = dataset_train["Fare"][dataset_train["Pclass"] == 3 ].std()
y3 = norm.pdf(x, loc=mean3, scale=std3) 
plt.plot(x,y3, color = "blue", label="Pclass 3")

mean2 = dataset_train["Fare"][dataset_train["Pclass"] == 2 ].mean()
std2 = dataset_train["Fare"][dataset_train["Pclass"] == 2 ].std()
y2 = norm.pdf(x, loc=mean2, scale=std2) 
plt.plot(x,y2, color = "red", label="Pclass 2")

mean1 = dataset_train["Fare"][dataset_train["Pclass"] == 1 ].mean()
std1 = dataset_train["Fare"][dataset_train["Pclass"] == 1 ].std()
y1 = norm.pdf(x, loc=mean1, scale=std1) 
plt.plot(x,y1, color = "green", label="Pclass 1")

plt.legend()
plt.show()

# %%
# Print the classes of Fare = 0:
workingset = dataset_train[dataset_train["Fare"] == 0 ] 
mean = [np.nan, dataset_train["Fare"][dataset_train["Pclass"] == 1 ].apply(lambda x: x if x>0 else np.nan).dropna().median() , dataset_train["Fare"][dataset_train["Pclass"] == 2 ].apply(lambda x: x if x>0 else np.nan).dropna().median(), dataset_train["Fare"][dataset_train["Pclass"] == 3 ].apply(lambda x: x if x>0 else np.nan).dropna().median()]
workingset["Fare"] = workingset["Pclass"].apply(lambda x: mean[x]) 

dataset_train.update(workingset)

# %%

# 1. extract title 
# 2. assume if it's mother/daughter
# 


# perhaps missing rows can be dropped? Let's analyze the test data:
dataset_train.info()
#  SibSp represents number of Brothers-Sisters or Wife-Husbands
#  Parch represents number of Parents or Kids 




#%%
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