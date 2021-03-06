import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, KFold
import tensorflow as tf
from sklearn.model_selection import train_test_split
from time import time

class ModelEvaluation:

    def __init__(self, clusters, x_cols, y_cols, simulations):
        self.clusters = clusters
        self.x_cols = x_cols
        self.y_cols = y_cols
        self.simulations = simulations
        self.results = self.createModels()

    def createModels(self):
        models = {"1":[], "2":[], "3":[], "4":[], "5":[]}
        c = 0
        for cluster in self.clusters:
            c += 1
            errors = {"Tree":[], "SVR":[], "RF":[], "ANN":[]}
            sigmas = {"Tree":[], "SVR":[], "RF":[], "ANN":[]}
            params = {"Tree":[], "SVR":[], "RF":[], "ANN":[]}
            for i in range(self.simulations):
                t0 = time()
                X_train, X_test, Y_train, Y_test = train_test_split(cluster[self.x_cols], cluster[self.y_cols], test_size=0.2)
                
                tree, tree_error, best_params = self.createDesitionTreeRegressor(X_train, Y_train, X_test, Y_test)
                errors["Tree"].append(tree_error)
                params["Tree"].append(tree_error)

                svr, svr_error, best_kernel = self.createSVR(X_train, Y_train, X_test, Y_test)
                errors["SVR"].append(svr_error)
                params["SVR"].append(svr_error)

                random_forest, random_forest_error, best_max_features = self.createRandomForestRegressor(X_train, Y_train, X_test, Y_test)
                errors["RF"].append(random_forest_error)
                params["RF"].append(random_forest_error)

                ann, ann_error, best_layers = self.createANNR(X_train, Y_train, X_test, Y_test)
                errors["ANN"].append(ann_error)
                params["ANN"].append(ann_error)
                
                tf = time()
                print(f"Quedan {((tf - t0)/60) * (10000 - i)} minutos.")

            for model_name in list(errors.keys()):
                errors[model_name] = np.mean(errors[model_name])
                cluster_model = pd.Series(params[model_name])
                sigmas[model_name] = np.std(errors[model_name])
                params[model_name] = list(cluster_model.value_counts().index[list(cluster_model.value_counts() == cluster_model.value_counts().max())])
            
            models[str(c)].extend({"Parameters":params}, {"MSE": errors}, {"STD":sigmas})
        return models
            

    def createANNR(self, X_train, Y_train, X_test, Y_test):
        errors = {}
        for i in range(3, 10):
            ann = tf.keras.models.Sequential()
            ann.add(tf.keras.layers.Dense(units=i, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=i, activation='relu'))
            ann.add(tf.keras.layers.Dense(units=1))
            ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
            ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)   
            y_pred = ann.predict(X_test)     
            errors[int(i)] = mean_squared_error(Y_test, y_pred)
        best_layers = int(min(errors))
        ann = tf.keras.models.Sequential()
        ann.add(tf.keras.layers.Dense(units=best_layers, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=best_layers, activation='relu'))
        ann.add(tf.keras.layers.Dense(units=1))
        ann.compile(optimizer = 'adam', loss = 'mean_squared_error')
        ann.fit(X_train, Y_train, batch_size = 32, epochs = 100)
        y_pred = ann.predict(X_test)     
        error = mean_squared_error(Y_test, y_pred)  
        return ann, error, best_layers

    
    def createDesitionTreeRegressor(self, X_train, Y_train, X_test, Y_test):
        criterions = ['mse', 'friedman_mse', 'mae']
        splitters = ['best', 'random']
        max_features_list = ['auto', 'sqrt', 'log2']
        errors = {}
        for criterion in criterions:
            for splitter in splitters:
                for max_feature in max_features_list:
                    tree = DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_features=max_feature).fit(X_train, Y_train)
                    y_pred = tree.predict(X_test)
                    errors[f"{criterion} {splitter} {max_feature}"] = mean_squared_error(Y_test, y_pred)
        
        best_params = min(errors)
        tree = DecisionTreeRegressor(criterion= best_params.split(' ')[0], splitter=best_params.split(' ')[1], max_features=best_params.split(' ')[2]).fit(X_train, Y_train)
        y_pred = tree.predict(X_test)
        error = mean_squared_error(Y_test, y_pred)
        return tree, error, best_params
                


    def createRandomForestRegressor(self, X_train, Y_train, X_test, Y_test):
        max_features_list = ['auto', 'sqrt', 'log2']
        errors = {}
        for max_feature in max_features_list:
            random_forest = RandomForestRegressor(max_features=max_feature).fit(X_train, Y_train)
            y_pred = random_forest.predict(X_test)
            errors[max_feature] = mean_squared_error(Y_test, y_pred)
        best_max_features = min(errors)
        random_forest = RandomForestRegressor(max_features=best_max_features).fit(X_train, Y_train)
        y_pred = random_forest.predict(X_test)
        error = mean_squared_error(Y_test, y_pred)
        return random_forest, error, best_max_features


    def createSVR(self, X_train, Y_train, X_test, Y_test):
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        errors = {}
        for kernel in kernels:
            svr = SVR( kernel = kernel).fit(X_train, Y_train)
            y_pred = svr.predict(X_test)
            errors[kernel] = mean_squared_error(Y_test, y_pred)
        best_kernel = min(errors)
        svr = SVR(kernel = best_kernel).fit(X_train, Y_train)
        y_pred = svr.predict(X_test)
        error = mean_squared_error(Y_test, y_pred)
        return svr, error, best_kernel        

