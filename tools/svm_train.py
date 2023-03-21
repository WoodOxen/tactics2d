import sys

from sklearn.model_selection import GridSearchCV, train_test_split
sys.path.append(".")
sys.path.append("..")
import logging
logging.basicConfig(level=logging.DEBUG)
import numpy as np
import pickle
import os

from sklearn import svm
import joblib
from tactics2d.participant.element import Vehicle, Pedestrian, Cyclist


"""
pedestrians and cyclist data file name:
"inD0-20_data_pedestrian_and_cycle.pkl"
"inD21-32_data_pedestrian_and_cycle.pkl"
"rounD0-23_data_pedestrian_and_cycle.pkl"
"uniD0-6_data_pedestrian_and_cycle.pkl"
"uniD7-12_data_pedestrian_and_cycle.pkl"
You should run levelX_parser_pedestrian_and_cyclist.py first to save these data.
"""

def get_features_from_trajectory(file_name:str, svm_X:np.ndarray, svm_y:np.ndarray, start_id:int):
    with open(file_name, "rb") as tf:
        data = pickle.load(tf)
    data_num = len(data)
    id = start_id
    for file_id in data.keys():
        
        # get label, 0 for pedestrain and 1 for cyclist
        label = 0 if type(data[file_id]) == Pedestrian else 1

        # calculate average speed
        states_num = len(data[file_id].trajectory.history_states)
        speed_average = data[file_id].trajectory.average_speed

        # calculate speed variance, accel average , speed max
        speeddif_sum = 0
        accel_sum = 0
        for state in data[file_id].trajectory.history_states.values():
            speeddif_sum += (state.speed - speed_average)**2
            accel_sum += state.accel
        speed_variance = speeddif_sum / states_num
        accel_average = accel_sum / states_num

        speed_max = max(state.speed for state in data[file_id].trajectory.history_states.values())
        # print(label,"\t",speed_average,"\t",speed_variance,"\t",speed_max,"\t",accel_average)

        svm_X[id][0] = speed_average
        svm_X[id][1] = speed_variance
        svm_X[id][2] = speed_max
        # svm_X[id][3] = accel_average
        svm_y[id] = label
        id += 1
    
    return id
    
def save_svm_train_data():
    """
    process data and save train data.

    You should run levelX_parser_pedestrian_and_cyclist.py first and after that you can call this function.
    """

    # inD 33 files + rounD 24 files + uniD 13 files = 14202 data
    # inD 33 files + rounD 24 files = 5603 data
    X = np.zeros((14202, 3))
    y = np.zeros((14202,))
    #X = np.zeros((5603, 3))
    #y = np.zeros((5603,))
    id_tmp = get_features_from_trajectory("./pedestrian_and_cyclist_trajectory/inD0-20_data_pedestrian_and_cycle.pkl",X,y,0)
    print("inD0-20",id_tmp)
    id_tmp = get_features_from_trajectory("./pedestrian_and_cyclist_trajectory/inD21-32_data_pedestrian_and_cycle.pkl",X,y,id_tmp)
    print("inD21-32",id_tmp)
    id_tmp = get_features_from_trajectory("./pedestrian_and_cyclist_trajectory/rounD0-23_data_pedestrian_and_cycle.pkl",X,y,id_tmp)
    print("rounD0-23",id_tmp)
    id_tmp = get_features_from_trajectory("./pedestrian_and_cyclist_trajectory/uniD0-6_data_pedestrian_and_cycle.pkl",X,y,id_tmp)
    print("uniD0-6",id_tmp)
    id_tmp = get_features_from_trajectory("./pedestrian_and_cyclist_trajectory/uniD7-12_data_pedestrian_and_cycle.pkl",X,y,id_tmp)
    print("uniD6-12",id_tmp)
    
    with open("./train_data/svm(no_accel)_trainData_features.pkl", "wb") as tf:
        pickle.dump(X,tf)
    with open("./train_data/svm(no_accel)_trainData_labels.pkl", "wb") as tf:
        pickle.dump(y,tf)


def svm_train_pedestrian_cyclist():
    """
    train the svm and save the svm model

    you should run save_svm_train_data() first to save the train data
    """
    
    logging.disable(logging.DEBUG)
    with open("./train_data/svm(no_accel)_trainData_features.pkl", "rb") as tf:
        X = pickle.load(tf)
    with open("./train_data/svm(no_accel)_trainData_labels.pkl", "rb") as tf:
        y = pickle.load(tf)
    
    
    grid = GridSearchCV(svm.SVC(), param_grid={"C":[0.1, 1, 10,100,1000], "gamma": [2.5,1, 0.2,0.025,0.01,0.002]}, cv=4)
    grid.fit(X, y)
    best_C = grid.best_params_['C']
    best_gamma = grid.best_params_['gamma']
    print("The best parameters are C : %f , gamma : %f with a score of %0.2f"% (best_C, best_gamma, grid.best_score_)) 

    clf = svm.SVC(kernel="rbf", C=0.1, gamma = 2.5)
    #clf = svm.SVC(kernel="linear", C=100)
    #clf = svm.SVC(kernel="poly", C=100)
    #clf = svm.SVC(kernel="rbf", C=best_C, gamma = best_gamma)
    #clf = svm.NuSVC(nu = 0.3,gamma="auto")
    
    for i in range(0,10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
        
        clf.fit(X_train, y_train)
        
        X_test_predict = clf.predict(X_test)
        true_num = np.sum(X_test_predict == y_test)
        print("Accuracy rate:",true_num/len(y_test))
    
    # save_path = os.path.abspath(os.path.join(os.path.dirname("__file__"),os.path.pardir))
    # save the model
    clf.fit(X, y)
    joblib.dump(clf, "svm_model2.m")

if __name__ == '__main__':
    # process and save the train data
    save_svm_train_data()

    # train the svm
    # you should run save_svm_train_data() first to save the train data
    # svm_train_pedestrian_cyclist()
