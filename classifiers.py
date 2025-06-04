from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from matplotlib import colormaps
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class Classifiers():
    def __init__(self,data):
        ''' 
        TODO: Write code to convert the given pandas dataframe into training and testing data 
        # all the data should be nxd arrays where n is the number of samples and d is the dimension of the data
        # all the labels should be nx1 vectors with binary labels in each entry 
        '''
        nxd = data.iloc[:, :-1].to_numpy()
        nx1 = data.iloc[:,-1].to_numpy()

        data.plot(kind='scatter', x='A',y='B', c='label', colormap='coolwarm')
        plt.savefig('initialdata.png')

        self.training_data, self.testing_data, self.training_labels, self.testing_labels = train_test_split(
            nxd, nx1, test_size=0.4)
        self.outputs = []
    
    def test_clf(self, clf, classifier_name=''):
        # TODO: Fit the classifier and extrach the best score, training score and parameters
        
        clf.fit(self.training_data, self.training_labels)

        training_score = cross_val_score(clf, self.training_data, self.training_labels, cv=5, scoring='accuracy').mean()
        testing_score = cross_val_score(clf, self.testing_data, self.testing_labels, cv=5, scoring='accuracy').mean()

        self.outputs.append(f"{classifier_name}, {training_score:.4f}, {testing_score:.4f}")

        self.plot(self.testing_data, clf.predict(self.testing_data),model=clf,classifier_name=classifier_name)

    def classifyNearestNeighbors(self):
        param_grid = {
            'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            'leaf_size': [5, 10, 15, 20, 25, 30]
        }

        nn = KNeighborsClassifier()
        grid_search = GridSearchCV(nn, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.training_data, self.training_labels)
        best_nn = grid_search.best_estimator_
        print(grid_search.best_params_)
        self.test_clf(best_nn, classifier_name='NN')
        
    def classifyLogisticRegression(self):
        param_grid = {
            'C': [0.1, 0.5, 1, 5, 10, 50, 100]
        }
        lg = LogisticRegression()
        grid_search = GridSearchCV(lg, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.training_data, self.training_labels)
        best_lg = grid_search.best_estimator_
        print(grid_search.best_params_)
        self.test_clf(best_lg, classifier_name='Logistic Regression')
    
    def classifyDecisionTree(self):
        temp=[]
        for x in range(1,51):
            temp.append(x)
        param_grid = {
            'max_depth': temp,
            'min_samples_split' : [2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        dt = DecisionTreeClassifier()
        grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.training_data, self.training_labels)
        best_dt = grid_search.best_estimator_
        print(grid_search.best_params_)
        self.test_clf(best_dt, classifier_name='Decision Tree')

    def classifyRandomForest(self):
        param_grid = {
            'max_depth': [1, 2, 3, 4, 5],
            'min_samples_split' : [2, 3, 4, 5, 6, 7, 8, 9, 10]
        }
        rf = RandomForestClassifier()
        grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.training_data, self.training_labels)
        best_rf = grid_search.best_estimator_
        print(grid_search.best_params_)
        self.test_clf(best_rf, classifier_name='Random Forest')

    def classifyAdaBoost(self):
        param_grid = {
            'n_estimators':[10, 20, 30, 40, 50, 60, 70]
        }
        ada = AdaBoostClassifier()
        grid_search = GridSearchCV(ada, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(self.training_data, self.training_labels)
        best_ada = grid_search.best_estimator_
        print(grid_search.best_params_)
        self.test_clf(best_ada, classifier_name='AdaBoost')

    def plot(self, X, Y, model,classifier_name = ''):
        X1 = X[:, 0]
        X2 = X[:, 1]

        X1_min, X1_max = min(X1) - 0.5, max(X1) + 0.5
        X2_min, X2_max = min(X2) - 0.5, max(X2) + 0.5

        X1_inc = (X1_max - X1_min) / 200.
        X2_inc = (X2_max - X2_min) / 200.

        X1_surf = np.arange(X1_min, X1_max, X1_inc)
        X2_surf = np.arange(X2_min, X2_max, X2_inc)
        X1_surf, X2_surf = np.meshgrid(X1_surf, X2_surf)

        L_surf = model.predict(np.c_[X1_surf.ravel(), X2_surf.ravel()])
        L_surf = L_surf.reshape(X1_surf.shape)

        plt.title(classifier_name)
        plt.contourf(X1_surf, X2_surf, L_surf, cmap = plt.cm.coolwarm, zorder = 1)
        plt.scatter(X1, X2, s = 38, c = Y)

        plt.margins(0.0)
        # uncomment the following line to save images
        plt.savefig(f'{classifier_name}.png')
        plt.show()

    
if __name__ == "__main__":
    df = pd.read_csv('input.csv')
    models = Classifiers(df)
    print('Classifying with NN...')
    models.classifyNearestNeighbors()
    print('Classifying with Logistic Regression...')
    models.classifyLogisticRegression()
    print('Classifying with Decision Tree...')
    models.classifyDecisionTree()
    print('Classifying with Random Forest...')
    models.classifyRandomForest()
    print('Classifying with AdaBoost...')
    models.classifyAdaBoost()

    with open("output.csv", "w") as f:
        print('Name, Best Training Score, Testing Score',file=f)
        for line in models.outputs:
            print(line, file=f)