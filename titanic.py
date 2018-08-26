import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

from sklearn.svm import SVC


from sklearn.model_selection import GridSearchCV
from time import time as time

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import cross_val_score

def load_data():
	#train_file_location = input("Enter the training set file location")
	#test_file_location = input("Enter the test set file location")
	#label_file_location = input("Enter the labels file location")
	
	train_file_location = "C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\Titanic\\all\\train.csv"
	test_file_location = "C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\Titanic\\all\\test.csv"
	label_file_location = "C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\Titanic\\all\\gender_submission.csv"
	
	train_data = pd.read_csv(train_file_location, 
		dtype = None,
		index_col = ["PassengerId"],
		delimiter= ',', 
		skiprows=1, 
		names = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
		converters= {5: lambda s: float(s or 0), 9: lambda j: round(float(j),2), 10: lambda j: str(j or "XXX"), 11: lambda e: str(e or "X")})
	train_data["Embarked_encoded"] = LabelEncoder().fit_transform(train_data["Embarked"])
	train_data["Sex_encoded"] = LabelEncoder().fit_transform(train_data["Sex"])

	train_X = train_data[["Pclass","Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Embarked_encoded", "Sex_encoded"]]
	train_Y = train_data[["Survived"]]
	
	test_data = pd.read_csv(test_file_location,
		dtype = None, 
		index_col = ["PassengerId"],
		delimiter= ',', 
		skiprows=1,
		names = ["PassengerId", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"], 
		converters= {4: lambda s: float(s or 0), 8: lambda j: round(float(j or 0), 2), 9: lambda j: str(j or "XXX"), 10: lambda e: str(e or "X")})
	test_data["Embarked_encoded"] = LabelEncoder().fit_transform(test_data["Embarked"])
	test_data["Sex_encoded"] = LabelEncoder().fit_transform(test_data["Sex"])

	test_X = test_data[["Pclass","Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked", "Embarked_encoded", "Sex_encoded"]]

	return train_X, train_Y, test_X, test_data.index.tolist()

def visualize_data(train_X, train_Y):
	record_count = train_X.shape[0]

	#Age validity
	invalid_age = train_X[train_X["Age"] == 0.0].shape[0]
	valid_age = record_count - invalid_age
	age_y = np.arange(2)
	age_validity = [valid_age, invalid_age]
	print('\n######################\nValid Age\n****** Total= {0} \nValid = {1} \nInvalid = {2}'.format(record_count, valid_age, invalid_age))
	plt.subplot(1,2,1)
	plt.bar(age_y, age_validity, align='center')
	plt.xticks(age_y, ["Valid", "Invalid"])
	plt.ylabel('Count')
	plt.title('Age validity')

	#Cabin validity
	invalid_cabin = train_X[train_X["Cabin"] == "XXX"].shape[0]
	valid_cabin = record_count - invalid_cabin
	cabin_y = np.arange(2)
	print('\n######################\nValid cabin\n******\nTotal= {0} \nValid = {1} \nInvalid = {2}'.format(record_count, valid_cabin, invalid_cabin))
	plt.subplot(1,2,2)
	cabin_validity = [valid_cabin, invalid_cabin]
	plt.bar(cabin_y, cabin_validity, align='center')
	plt.xticks(cabin_y, ["Valid", "Invalid"])
	plt.ylabel('Count')
	plt.title('Cabin validity')

	plt.show()

	#Age Vs Survival
	valid_age_df = train_X[train_X["Age"] != 0.0]
	max_age = valid_age_df["Age"].max()
	min_age = valid_age_df["Age"].min()
	range_age = max_age - min_age
	print('\n######################\nAge ranges\n******\nRange = {0} \nMin age = {1} \nMax age = {2}'.format(range_age, min_age, max_age))
	plt.plot(valid_age_df["Age"], train_Y.loc[valid_age_df.index]["Survived"], 'ro')
	plt.title('Age and survival relevance')

	plt.show()

	#Gender - Age Vs Survival
	#Female
	female_age_df = valid_age_df[valid_age_df["Sex"] == 'female']
	female_survived = train_Y.loc[female_age_df.index][train_Y.loc[female_age_df.index]["Survived"] == 1].shape[0]
	female_not_survived = female_age_df.shape[0] - female_survived
	print('\n######################\nFemale Survival\n******\nTotal= {0} \nSurvived = {1} \nNot survived = {2}'.format(female_age_df.shape[0], female_survived, female_not_survived))
	plt.bar(np.arange(2), [female_survived, female_not_survived], align='center')
	plt.xticks(np.arange(2), ["Survived", "Not survived"])
	plt.ylabel('Count')
	plt.title('Female - Age and survival relevance')
	plt.show()
	#Male
	male_age_df = valid_age_df[valid_age_df["Sex"] == 'male']
	male_survived = train_Y.loc[male_age_df.index][train_Y.loc[male_age_df.index]["Survived"] == 1].shape[0]
	male_not_survived = male_age_df.shape[0] - male_survived
	print('\n######################\nMale Survival\n******\nTotal= {0} \nSurvived = {1} \nNot survived = {2}'.format(male_age_df.shape[0], male_survived, male_not_survived))
	plt.bar(np.arange(2), [male_survived, male_not_survived], align='center')
	plt.xticks(np.arange(2), ["Survived", "Not survived"])
	plt.ylabel('Count')
	plt.title('Male - Age and survival relevance')
	plt.show()

def data_preprocessing(train_X, test_X):
	#Dropping column Cabin
	train_X.drop(["Cabin"], axis = 1, inplace = True)
	test_X.drop(["Cabin"], axis = 1, inplace = True)

	#Female - Assigning mean age to invalid age entries
	train_X_female = train_X[train_X["Sex"] == 'female']
	train_X_valid_age_female = train_X_female[train_X_female["Age"] != 0.0]
	train_X_invalid_age_female = train_X_female[train_X_female["Age"] == 0.0]
	avg_female_age = round(train_X_valid_age_female["Age"].mean(), 1)
	print('\n######################\nAverage female age: {0}'.format(avg_female_age))
	train_X.ix[train_X_invalid_age_female.index, "Age"] = avg_female_age

	#Male - Assigning mean age to invalid age entries
	train_X_male = train_X[train_X["Sex"] == 'male']
	train_X_valid_age_male = train_X_male[train_X_male["Age"] != 0.0]
	train_X_invalid_age_male = train_X_male[train_X_male["Age"] == 0.0]
	avg_male_age = round(train_X_valid_age_male["Age"].mean(), 1)
	print('\n######################\nAverage male age: {0}'.format(avg_male_age))
	train_X.ix[train_X_invalid_age_male.index, "Age"] = avg_male_age

	#Assigning mean age to invalid age entries in test data
	test_X_female = train_X[train_X["Sex"] == 'female']
	test_X_invalid_age_female = test_X_female[test_X_female["Age"] == 0.0]
	test_X.ix[test_X_invalid_age_female.index, "Age"] = avg_female_age
	test_X_male = train_X[train_X["Sex"] == 'male']
	test_X_invalid_age_male = test_X_male[test_X_male["Age"] == 0.0]
	test_X.ix[test_X_invalid_age_male.index, "Age"] = avg_male_age

	return train_X, test_X

def random_forest_classifier(train_X, train_Y):
	n_estimators_range = np.arange(1,31)
	max_depth_range = np.arange(1, 51)
	param_grid = dict(n_estimators = n_estimators_range, max_depth = max_depth_range)
	grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring = 'accuracy', cv=5, n_jobs = -1)
	grid.fit(train_X, train_Y.values.ravel())
	print(grid.best_params_)
	print(grid.best_score_)
	classifier = grid.best_estimator_
	
	func_name = random_forest_classifier.__name__
	return classifier, func_name

def adaboost_classifier(train_X, train_Y):
	DTC = DecisionTreeClassifier(criterion = 'entropy', max_depth = 48, min_samples_split = 13, splitter = 'random', random_state = 11)
	ABC = AdaBoostClassifier(base_estimator = DTC)
	n_estimators_range = np.arange(1,31)
	learning_rate_range = np.arange(0.1, 1, 0.01)
	param_grid = dict(learning_rate = learning_rate_range, n_estimators = n_estimators_range) 
	grid = GridSearchCV(ABC, param_grid, scoring = 'accuracy', cv=5, n_jobs = -1)
	grid.fit(train_X, train_Y.values.ravel())
	print(grid.best_params_)
	print(grid.best_score_)
	classifier = grid.best_estimator_

	func_name = adaboost_classifier.__name__
	return classifier, func_name

def decision_tree_classifier(train_X, train_Y):
	criterion_range = ('gini', 'entropy')
	splitter_range = ('best', 'random')
	max_depth_range = np.arange(1, 51)
	min_samples_split_range = np.arange(2, 26)
	#max_features_range = ('auto', 'sqrt', 'log2', 'auto')
	param_grid = dict(max_depth = max_depth_range, criterion = criterion_range, splitter = splitter_range, min_samples_split = min_samples_split_range)
	grid = GridSearchCV(DecisionTreeClassifier(), param_grid, scoring = 'accuracy', cv=5, n_jobs = -1)
	grid.fit(train_X, train_Y.values.ravel())
	print(grid.best_params_)
	print(grid.best_score_)
	classifier = grid.best_estimator_

	func_name = decision_tree_classifier.__name__
	return classifier, func_name

def knn_classifier(train_X, train_Y):
	
	n_neighbors_range = np.arange(1, 51)
	algorithm_range = ('auto', 'ball_tree', 'kd_tree', 'brute')
	param_grid = dict(n_neighbors = n_neighbors_range, algorithm = algorithm_range)
	grid = GridSearchCV(KNeighborsClassifier(), param_grid, scoring = 'accuracy', cv = 6, n_jobs = -1)
	grid.fit(train_X, train_Y.values.ravel())
	print(grid.best_params_)
	print(grid.best_score_)
	classifier = grid.best_estimator_

	func_name = knn_classifier.__name__
	return classifier, func_name

def svc_classifier(train_X, train_Y):
	Cs = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
	gammas = [0.001, 0.01, 0.1, 1]
	kernels = ('rbf', 'linear')
	param_grid = {'C': Cs, 'gamma' : gammas, 'kernel' : kernels}
	grid = GridSearchCV(SVC(), param_grid, scoring = 'accuracy', cv = 6, n_jobs = -1)
	grid.fit(train_X, train_Y.values.ravel())
	print(grid.best_params_)
	print(grid.best_score_)
	classifier = grid.best_estimator_

	func_name = knn_classifier.__name__
	return classifier, func_name

def naive_bayes_classifier(train_X, train_Y):
	classifier = GaussianNB()
	classifier.fit(train_X, train_Y)
	scores = cross_val_score(classifier, train_X, train_Y.values.ravel(), cv=5)
	print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

	func_name = naive_bayes_classifier.__name__
	return classifier, func_name


def classifier_predict(classifier, test_X,):
	predictions = classifier.predict(test_X)
	return predictions	

def main():
	
	train_X, train_Y, test_X, test_id = load_data()
	train_X, test_X = data_preprocessing(train_X, test_X)

	classifier_train_X = train_X[["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked_encoded", "Sex_encoded"]]
	classifier_test_X = test_X[["Pclass", "Age", "SibSp", "Parch", "Fare", "Embarked_encoded", "Sex_encoded"]]
	classifier_train_X.to_csv("C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\Titanic\\all\\cleaned_train_X.csv", sep = ',')
	classifier_test_X.to_csv("C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\Titanic\\all\\cleaned_test_X.csv", sep = ',')

	time_start = time()
	classifier, func_name = svc_classifier(classifier_train_X, train_Y)
	predictions = classifier_predict(classifier, classifier_test_X)
	time_end = time()
	time_taken = time_end - time_start
	results_dict = {'PassengerId': test_id, 'Survived': predictions}
	results_df = pd.DataFrame(results_dict, index = results_dict['PassengerId'], columns = ["Survived"])
	results_df.index.name = 'PassengerId'
	file_name = "C:\\Users\\niranjan\\Documents\\Machine_Learning\\kaggle\\Titanic\\" + func_name.lower() + "_predictions.csv"
	results_df.to_csv(file_name, sep = ',')

	print('\n$$$$$$$$$$$$$\n{0}\n***********\nTime taken = {1}\n$$$$$$$$$$$$$\n'. format(func_name.upper(), time_taken))


if __name__== "__main__":
  main()