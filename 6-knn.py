# Week 6 - k-NN Assignment

import pandas as pd
import numpy as np

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float , 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int,
              'view':int}


train_set_src = pd.read_csv('data/kc_house_data_small_train.csv', dtype = dtype_dict)
test_set_src = pd.read_csv('data/kc_house_data_small_test.csv', dtype = dtype_dict)
val_set_src = pd.read_csv('data/kc_house_data_validation.csv', dtype = dtype_dict)


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


def normalize_features(feature_matrix):
    '''
        Normalizes columns of the matrix.
    '''
    norms = np.linalg.norm(feature_matrix, axis = 0)
    norm_mat = feature_matrix / norms

    return (norm_mat, norms)


feature_list = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view',
                'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated',
                'lat', 'long', 'sqft_living15', 'sqft_lot15']

(train_feat_set, train_output_set) = get_numpy_data(train_set_src, feature_list ,'price')
(test_feat_set, test_output_set) = get_numpy_data(test_set_src, feature_list,'price')
(val_feat_set, val_output_set) = get_numpy_data(val_set_src, feature_list,'price')

(train_feat_set_normalized, train_norms) = normalize_features(train_feat_set)
test_feat_set_normalized = test_feat_set / train_norms
val_feat_set_normalized = val_feat_set / train_norms







# Compute a single distance

print(test_feat_set_normalized[0])
print(train_feat_set_normalized[9])

def dist_calc(a,b,n):
    '''
        calculates the n-norm'ed distance metric between 2 points a, b
    '''

    return (np.sum((a - b) ** n)) ** (1/n)



print(dist_calc(test_feat_set_normalized[0], train_feat_set_normalized[9], 2))
# 0.059723594322325047

for i in range(10):
    print(dist_calc(test_feat_set_normalized[0], train_feat_set_normalized[i], 2))
# index = 8, so 9th house



diff = train_feat_set_normalized[:] - test_feat_set_normalized[0]
print(np.sum(diff ** 2, axis = 1)[15] / np.sum(diff[15] ** 2)) # almost alike

distances = np.sqrt(np.sum(diff ** 2, axis = 1))
print(distances[100])
# checker: should be equal 0.0237082324496 (ours 0237082324167 ... slight difference ??)



def compute_distances(features_instances, features_query):
    """
        Vectorized implementation of euclidean distance calculation
    """
    diff = features_instances[:] - features_query
    distances = np.sqrt(np.sum(diff ** 2, axis = 1))
    return distances


dist_vec = compute_distances(train_feat_set_normalized, test_feat_set_normalized[2])
print(dist_vec.argmin())
# 382



# predicted value of house
print(train_output_set[382])
# 249000.0







# k-NN Regression

def k_nearest_neighbors(k, feature_train, features_query):

    dist_vec = compute_distances(feature_train, features_query)
    sort_idx = np.argsort(dist_vec)
    dist_vec_first_k = dist_vec[sort_idx[:k]]
    sort_idx_first_k = sort_idx[:k]
    return dist_vec_first_k, sort_idx_first_k

# checker - using our above example, should return the 382th house - OK
# print(k_nearest_neighbors(1, train_feat_set_normalized, test_feat_set_normalized[2]))

print(k_nearest_neighbors(4, train_feat_set_normalized, test_feat_set_normalized[2]))
# 382, 1149, 4087, 3142

print(predict_output_of_query(4, train_feat_set_normalized,
                              train_output_set, test_feat_set_normalized[2]))
# 413987.5

# checker: 413987.5 - OK
# (train_output_set[382] + train_output_set[1149] + train_output_set[4087] + train_output_set[3142])/4



def predict_output_of_query(k, features_train, output_train, features_query):
    '''
        function to predict the value of every house in the query set
        where the query can either be vec (single query) or matrix (multiple queries)
    '''
    if features_query.ndim != 1:
        queries_cnt = features_query.shape[0]
        pred = np.zeros(queries_cnt)
        for i in range(queries_cnt):
                dist_vec = compute_distances(features_train, features_query[i,:])
                sort_idx = np.argsort(dist_vec)
                pred[i] =  np.average(output_train[sort_idx[:k]])

    else:
        dist_vec = compute_distances(features_train, features_query)
        sort_idx = np.argsort(dist_vec)
        pred =  np.average(output_train[sort_idx[:k]])

    return pred


# checker - using our above example, should return 249000 - OK
# predict_output_of_query(1, train_feat_set_normalized,
#                         train_output_set, test_feat_set_normalized[2] )

min_10_predict = predict_output_of_query(10, train_feat_set_normalized,
                        train_output_set, test_feat_set_normalized[0:10] )
print(min_10_predict.argmin())
# index = 6 (7th house), with predicted value 350032






# Choosing best k using validation

k_vals = np.linspace(1,15,15)

rss_vec = np.zeros(len(k_vals))
for i,k  in enumerate(k_vals):
    preds = predict_output_of_query(k, train_feat_set_normalized,
                                     train_output_set, val_feat_set_normalized)
    errors = val_output_set - preds
    rss_vec[i] = np.sum(errors ** 2)
print(rss_vec.argmin())
# index = 7, so k = 8

print(rss_vec[rss_vec.argmin()])
# 6.73616787e+13


errors = test_output_set - predict_output_of_query(8, train_feat_set_normalized,
                                     train_output_set, test_feat_set_normalized)
print(np.sum(errors **2))
# RSS on test set: 133118823551516








