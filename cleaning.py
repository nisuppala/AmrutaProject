import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

#Training (891 Entries) & Testing (417 Entries) data
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
all_data = [train_data, test_data]

print( train_data[["Pclass","Survived"]].groupby(["Pclass"], as_index = False).mean() )

print( train_data[["Sex","Survived"]].groupby(["Sex"], as_index = False).mean() )

print( train_data[["SibSp","Survived"]].groupby(["SibSp"], as_index = False).mean() )

print( train_data[["Parch","Survived"]].groupby(["Parch"], as_index = False).mean() )

for data in all_data:
    data['Embarked'] = data['Embarked'].fillna('S')
print( train_data[["Embarked","Survived"]].groupby(["Embarked"], as_index = False).mean() )

for data in all_data:
    data["Deck"] = data["Cabin"].str.slice(0, 1)
    data['Deck'] = data['Deck'].fillna('U')
print(train_data[["Deck", "Survived"]].groupby(["Deck"], as_index=False).mean())

for data in all_data:
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
train_data['category_fare'] = pd.qcut(train_data['Fare'], 4)
print( train_data[["category_fare","Survived"]].groupby(["category_fare"], as_index = False).mean() )

for data in all_data:
    age_avg  = data['Age'].mean()
    age_std  = data['Age'].std()
    age_null = data['Age'].isnull().sum()
    random_list = np.random.randint(age_avg - age_std, age_avg + age_std , size = age_null)
    data['Age'][np.isnan(data['Age'])] = random_list
    data['Age'] = data['Age'].astype(int)
train_data['category_age'] = pd.cut(train_data['Age'], 5)
print( train_data[["category_age","Survived"]].groupby(["category_age"], as_index = False).mean() )



for data in all_data:

    #Mapping Sex
    sex_map = {'female': 0, 'male': 1}
    data['Sex'] = data['Sex'].map(sex_map).astype(int)

    #Mapping Embarked
    embark_map = {'S': 0, 'C': 1, 'Q': 2}
    data['Embarked'] = data['Embarked'].map(embark_map).astype(int)

    # Mapping Deck
    deck_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'T': 7, 'U': 8}
    data['Deck'] = data['Deck'].map(deck_map).astype(int)

    #Mapping Fare
    data.loc[ data['Fare'] <= 7.91, 'Fare']                            = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare']                               = 3
    data['Fare'] = data['Fare'].astype(int)

    #Mapping Age
    data.loc[ data['Age'] <= 16, 'Age']                       = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age']                        = 4

#Feature Selection
drop_elements = ["Name", "Ticket", "Cabin"]

train_data = train_data.drop(drop_elements, axis = 1)
train_data = train_data.drop(['PassengerId','category_fare', 'category_age'], axis = 1)
test_data = test_data.drop(drop_elements, axis = 1)

#Print ready to use data
print(train_data.head(10))


X_train = train_data.drop("Survived", axis=1)
Y_train = train_data["Survived"]
X_test  = test_data.drop("PassengerId", axis=1).copy()


classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
accuracy = round(classifier.score(X_train, Y_train) * 100, 2)
print("Model Accuracy: ", accuracy)

import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()

