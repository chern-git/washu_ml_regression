# Week 4 - Ridge regression Assignment 1

import numpy as np
import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float,
              'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

train_data = pd.read_csv('data/kc_house_train_data.csv', dtype=dtype_dict)
test_data = pd.read_csv('data/kc_house_test_data.csv', dtype=dtype_dict)
sales = pd.read_csv('data/kc_house_data.csv', dtype=dtype_dict)

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


def feature_derivative_ridge(errors, feature, weight, l2_penalty, feature_is_constant):
    if (feature_is_constant == True):
        deriv_reg = 0
    else:
        deriv_reg = 2 * l2_penalty * weight
    deriv_RSS = 2 * np.sum(np.dot(errors, feature))
    deriv = deriv_RSS + deriv_reg
    return deriv


def ridge_regression_grad_desc(
        feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):

    weights = np.array(initial_weights, dtype = 'f')
    t = 0

    while t < max_iterations:
        pred = predict_output(feature_matrix, weights)
        errors = pred - output
        for i in range(len(weights)): # loop over each weight
            derivative = \
                feature_derivative_ridge(errors, feature_matrix[:,i], weights[i], l2_penalty, i == 0)
            weights[i] = weights[i] - (step_size * derivative)
        t = t + 1

    return weights


def ridge_regression_grad_desc_vec(
        feature_matrix, output, initial_weights, step_size, l2_penalty, max_iterations=100):
    '''
        vectorized implementation of gradient descent
    '''
    weights = np.array(initial_weights, dtype = 'f')
    t = 0

    while t < max_iterations:
        pred = predict_output(feature_matrix, weights)
        errors = output - pred
        weights = weights - step_size * \
                            feature_derivative_ridge_vec(errors, feature_matrix, weights, l2_penalty)
        t = t + 1
    return weights


def feature_derivative_ridge_vec(errors, feature, weight, l2_pen):
    """
        vectorized implementation to ridge step
    """
    grad = (-2 * np.dot(np.transpose(feature), errors))
    regul = 2 * l2_pen * weight

    # manually set the first element to remove regularization for intercept term if array
    if hasattr(weight,'__len__'):
        regul[0] = 0
    else:
        regul = 0
    return (grad + regul)

# Check values: OK
# (example_features, example_output) = get_numpy_data(sales, ['sqft_living'], 'price')
# my_weights = np.array([1., 10.])
# test_predictions = predict_output(example_features, my_weights)
# errors = test_predictions - example_output # prediction errors
# print(feature_derivative_ridge(errors, example_features[:,1], my_weights[1], 1, False))
# print(np.sum(errors*example_features[:,1])*2+20.)
# print(feature_derivative_ridge_vec(errors, example_features[:, 1], my_weights[1], 1))
# print(feature_derivative_ridge(errors, example_features[:,0], my_weights[0], 1, True))
# print(np.sum(errors)*2.)
# print(feature_derivative_ridge_vec(errors, example_features[:, 0], my_weights[0], 1))




(simple_feature_matrix, output) = get_numpy_data(train_data, ['sqft_living'], 'price')
(simple_test_feature_matrix, test_output) = get_numpy_data(test_data, ['sqft_living'], 'price')

step_size = 1e-12
max_iter = 1000
l2_penalty = 0
initial_weights = np.zeros(2)

simple_weights_0_penalty = ridge_regression_grad_desc(simple_feature_matrix, output,
                                                      initial_weights, step_size, l2_penalty, max_iter)
print(simple_weights_0_penalty)

simple_weights_0_penalty_2 = ridge_regression_grad_desc_vec(simple_feature_matrix, output,
                                                      initial_weights, step_size, l2_penalty, max_iter)
print(simple_weights_0_penalty_2)



# High regularization Case

l2_penalty = 1e11
simple_weights_high_penalty = ridge_regression_grad_desc(simple_feature_matrix, output,
                                                      initial_weights, step_size, l2_penalty, max_iter)
simple_weights_high_penalty_2 = ridge_regression_grad_desc_vec(simple_feature_matrix, output,
                                                      initial_weights, step_size, l2_penalty, max_iter)
print(simple_weights_high_penalty)
print(simple_weights_high_penalty_2)




# RSS on no regularization
print(np.sum((test_output -
    (simple_weights_0_penalty[0] * simple_test_feature_matrix[:,0] +
    simple_weights_0_penalty[1] * simple_test_feature_matrix[:,1])) ** 2))


# RSS on high regularization
print(np.sum((test_output -
    (simple_weights_high_penalty[0] * simple_test_feature_matrix[:,0] +
    simple_weights_high_penalty[1] * simple_test_feature_matrix[:,1])) ** 2))







# Part 2

# Initial weights

model_features = ['sqft_living', 'sqft_living15']
my_output = 'price'
(feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
(test_feature_matrix, test_output) = get_numpy_data(test_data, model_features, my_output)

step_size = 1e-12
max_iter = 1000
l2_penalty = 0
initial_weights = np.zeros(3)

multiple_weights_0_penalty = ridge_regression_grad_desc(feature_matrix, output,
                                                      initial_weights, step_size, l2_penalty, max_iter)
multiple_weights_0_penalty_2 = ridge_regression_grad_desc_vec(feature_matrix, output,
                                                      initial_weights, step_size, l2_penalty, max_iter)
print(multiple_weights_0_penalty)
print(multiple_weights_0_penalty_2)


l2_penalty = 1e11
multiple_weights_high_penalty = \
    ridge_regression_grad_desc(feature_matrix, output,
        initial_weights, step_size, l2_penalty, max_iter)
multiple_weights_high_penalty_2 = \
    ridge_regression_grad_desc_vec(feature_matrix, output,
        initial_weights, step_size, l2_penalty, max_iter)
print(multiple_weights_high_penalty)
print(multiple_weights_high_penalty_2)


# RSS on no regularization
print(np.sum((test_output - (multiple_weights_0_penalty[0] * test_feature_matrix[:,0] +
    multiple_weights_0_penalty[1] * test_feature_matrix[:,1] +
        multiple_weights_0_penalty[2] * test_feature_matrix[:,2])) ** 2))



# RSS on high regularization
print(np.sum((test_output - (multiple_weights_high_penalty[0] * test_feature_matrix[:,0] +
    multiple_weights_high_penalty[1] * test_feature_matrix[:,1] +
        multiple_weights_high_penalty[2] * test_feature_matrix[:,2])) ** 2))







# Prediction task


error_no_reg = test_output[0] - (multiple_weights_0_penalty[0] * test_feature_matrix[0,0] +
    multiple_weights_0_penalty[1] * test_feature_matrix[0,1] +
        multiple_weights_0_penalty[2] * test_feature_matrix[0,2])
print(error_no_reg)



error_hi_reg = test_output[0] - (multiple_weights_high_penalty[0] * test_feature_matrix[0,0] +
    multiple_weights_high_penalty[1] * test_feature_matrix[0,1] +
        multiple_weights_high_penalty[2] * test_feature_matrix[0,2])
print(error_hi_reg)