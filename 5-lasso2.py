# Week 5 - LASSO Assignment 2

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int,
              'view':int}


def get_numpy_data(df, features, predictor):
    '''
    :param df: data frame to extract features from
    :param features: list of features to extract from the data frame
    :return: returns a feature matrix consisting of a first columns of ones, followed by columns
     containing the values of the input features in the data set in the same order as the input
     list.
    '''
    ones = np.ones([df.shape[0],1])
    output = pd.DataFrame(df, columns = features)
    output = np.concatenate([ones, output], axis = 1)
    return [output, df[predictor]]


def predict_output(feature_mat, weights):
    '''
    This function accepts a 2D array ‘feature_matrix’ and a 1D array ‘weights’
    and return a 1D array ‘predictions’.
    '''
    return np.dot(feature_mat,weights)


def normalize_features(feature_matrix):
    '''
        Normalizes columns of the matrix.
    '''
    norms = np.linalg.norm(feature_matrix, axis = 0)
    norm_mat = feature_matrix / norms

    return (norm_mat, norms)





sales = pd.read_csv('data/kc_house_data.csv', dtype=dtype_dict)

(simple_set, output) = get_numpy_data(sales, ['sqft_living', 'bedrooms'], 'price')
simple_set_normalized, simple_set_norms = normalize_features(simple_set)

initial_weights = [1,4,1]

pred = predict_output(simple_set_normalized, initial_weights)


def lasso_coordinate_descent_step(i, feature_matrix, output, initial_weights, l1_penalty):
    '''
        produces a single step in coordinate descent for LASSO
    '''

    weights = np.array(initial_weights, dtype = 'f')
    pred = predict_output(feature_matrix, initial_weights)
    error = output - pred
    ro_i = np.sum(np.dot(feature_matrix[:,i],
        error + np.dot(weights[i], feature_matrix[:,i])))

    if i == 0:
        new_weight_i = ro_i
    elif ro_i < -l1_penalty/2:
        new_weight_i = ro_i + l1_penalty/2
    elif ro_i > l1_penalty/2:
        new_weight_i = ro_i - l1_penalty/2
    else:
        new_weight_i = 0

    return new_weight_i

# function test - OK
# import math
# print(lasso_coordinate_descent_step(1, np.array([[3./math.sqrt(13),1./math.sqrt(10)],
#     [2./math.sqrt(13),3./math.sqrt(10)]]), np.array([1., 1.]), np.array([1., 4.]), 0.1))



ro = np.zeros(3)
for i in range(3):
    ro[i] = np.sum(simple_set_normalized[:,i] * \
            (output - pred + (initial_weights[i] * simple_set_normalized[:,i])))
print(ro * 2)


def lasso_cyclical_coordinate_descent(
    feature_matrix, output, initial_weights, l1_penalty, tolerance):
    '''
        produces the complete coordinate descent for LASSO
    '''
    weights = np.array(initial_weights, dtype = 'f')

    # first compulsory step
    for i in range(len(initial_weights)):
        weights[i] = lasso_coordinate_descent_step(
            i, feature_matrix, output, initial_weights, l1_penalty)
    weights_chg = weights

    while max(weights_chg) >= tolerance:
        weights_old = weights.copy()
        for i in range(len(initial_weights)):
            weights[i] = lasso_coordinate_descent_step(
                i, feature_matrix, output, weights, l1_penalty)
        weights_chg = np.abs(weights - weights_old)

    return weights


(simple_set, output) = get_numpy_data(sales, ['sqft_living', 'bedrooms'], 'price')
(simple_set_normalized, simple_norms) = normalize_features(simple_set)
initial_weights = np.zeros(3)
l1_penalty = 1e7
tol = 1.0

weights = lasso_cyclical_coordinate_descent(
    simple_set_normalized, output, initial_weights, l1_penalty, tol)
print(weights)

rss = np.sum((output - np.dot(weights, np.transpose(simple_set_normalized))) **2)
print(rss)







train_set = pd.read_csv('data/kc_house_train_data.csv', dtype=dtype_dict)
test_set = pd.read_csv('data/kc_house_test_data.csv', dtype=dtype_dict)

features_to_use = ['bedrooms', 'bathrooms', 'sqft_living' , 'sqft_lot', 'floors',
                   'waterfront', 'view', 'condition', 'grade', 'sqft_above',
                   'sqft_basement', 'yr_built', 'yr_renovated']

train_set['floors'] = train_set['floors'].astype('f')
test_set['floors'] = test_set['floors'].astype('f')

(multi_train_set, multi_train_output) = get_numpy_data(train_set, features_to_use, 'price')

(multi_train_set_normalized, multi_train_norms) = normalize_features(multi_train_set)

initial_weights = np.zeros(14)
l1_penalty = 1e7
tol = 1

weights_1e7 = lasso_cyclical_coordinate_descent(
    multi_train_set_normalized, multi_train_output, initial_weights, l1_penalty, tol)
for i in range(len(features_to_use)):
    print(features_to_use[i], " ", weights_1e7[i+1])
# we have a non-zero intercept as well



initial_weights = np.zeros(14)
l1_penalty = 1e8
tol = 1

weights_1e8 = lasso_cyclical_coordinate_descent(
    multi_train_set_normalized, multi_train_output, initial_weights, l1_penalty, tol)
print(weights_1e8)



initial_weights = np.zeros(14)
l1_penalty = 1e4
tol = 5e5

weights_1e4 = lasso_cyclical_coordinate_descent(
    multi_train_set_normalized, multi_train_output, initial_weights, l1_penalty, tol)
for i in range(len(features_to_use)):
    print(features_to_use[i], " ", weights_1e4[i+1])
# we have a non-zero intercept as well






# Evaluating learned models on test data

(multi_test_set, multi_test_output) = get_numpy_data(test_set, features_to_use, 'price')

weights_1e7_norm = weights_1e7 / multi_train_norms
weights_1e8_norm = weights_1e8 / multi_train_norms
weights_1e4_norm = weights_1e4 / multi_train_norms

print("Model 1e7: ", np.sum((multi_test_output - np.dot(multi_test_set, weights_1e7_norm)) ** 2))
print("Model 1e8: ", np.sum((multi_test_output - np.dot(multi_test_set, weights_1e8_norm)) ** 2))
print("Model 1e4: ", np.sum((multi_test_output - np.dot(multi_test_set, weights_1e4_norm)) ** 2))

# model 1e4 has the lowest RSS