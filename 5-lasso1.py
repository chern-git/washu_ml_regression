# Week 5 - LASSO Assignment 1

import pandas as pd
import numpy as np


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str,
              'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':float,
              'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int,
              'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('data/kc_house_data.csv', dtype=dtype_dict)

from math import log, sqrt
sales['sqft_living_sqrt'] = sales['sqft_living'].apply(sqrt)
sales['sqft_lot_sqrt'] = sales['sqft_lot'].apply(sqrt)
sales['bedrooms_square'] = sales['bedrooms']*sales['bedrooms']
sales['floors_square'] = sales['floors']*sales['floors']

l1_penalty = 5e2

features_to_use = ['bedrooms', 'bedrooms_square', 'bathrooms', 'sqft_living', 'sqft_living_sqrt',
            'sqft_lot', 'sqft_lot_sqrt', 'floors', 'floors_square', 'waterfront', 'view',
            'condition', 'grade', 'sqft_above','sqft_basement', 'yr_built', 'yr_renovated']

from sklearn import linear_model
model_all = linear_model.Lasso(alpha = l1_penalty, normalize= True)
model_all.fit(sales[features_to_use], sales['price'])


def print_coef(model, predictors):
    '''
        custom function to print coefficients of LASSO model
    '''
    coef_vec = list(model.coef_)
    print("%-16s" % "Predictor", "| ", "Coefficient")
    print("-----------------------------------------")
    for i,p in enumerate(coef_vec):
        print("%-16s" % predictors[i], ": ", p)

print_coef(model_all, features_to_use)







testing = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)
training = pd.read_csv('data/wk3_kc_house_train_data.csv', dtype=dtype_dict)
validation = pd.read_csv('data/wk3_kc_house_valid_data.csv', dtype=dtype_dict)

testing['sqft_living_sqrt'] = testing['sqft_living'].apply(sqrt)
testing['sqft_lot_sqrt'] = testing['sqft_lot'].apply(sqrt)
testing['bedrooms_square'] = testing['bedrooms']*testing['bedrooms']
testing['floors_square'] = testing['floors']*testing['floors']

training['sqft_living_sqrt'] = training['sqft_living'].apply(sqrt)
training['sqft_lot_sqrt'] = training['sqft_lot'].apply(sqrt)
training['bedrooms_square'] = training['bedrooms']*training['bedrooms']
training['floors_square'] = training['floors']*training['floors']

validation['sqft_living_sqrt'] = validation['sqft_living'].apply(sqrt)
validation['sqft_lot_sqrt'] = validation['sqft_lot'].apply(sqrt)
validation['bedrooms_square'] = validation['bedrooms']*validation['bedrooms']
validation['floors_square'] = validation['floors']*validation['floors']



l1_pen_vec = np.logspace(1, 7, num=13)
rss_vec = np.zeros(len(l1_pen_vec))

for i,p in enumerate(l1_pen_vec):
    model = linear_model.Lasso(alpha = p, normalize= True)
    model.fit(training[features_to_use], training['price'])
    errors = model.predict(validation[features_to_use]) - validation['price']
    rss_vec[i] = np.sum(errors ** 2)

print(l1_pen_vec[rss_vec == min(rss_vec)])
# 10 is the l1 penalty that gives the lowest RSS


l1_pen_to_use = l1_pen_vec[(rss_vec == min(rss_vec))]
model = linear_model.Lasso(alpha = l1_pen_to_use, normalize= True)
model.fit(training[features_to_use], training['price'])
errors = model.predict(testing[features_to_use]) - testing['price']
rss_test = np.sum(errors ** 2)

print(np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_))
# number of nonzero weights (including intercept) = 15






max_nonzeros = 7
l1_pen_vec = np.logspace(1, 4, num=20)
nonzero_weights_cnt = np.zeros(len(l1_pen_vec))

for i,p in enumerate(l1_pen_vec):
    model = linear_model.Lasso(alpha = p, normalize= True)
    model.fit(training[features_to_use], training['price'])
    nonzero_weights_cnt[i] = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)

    # any smaller than pen_min will give us more non-zeros than required
    l1_pen_min = max(l1_pen_vec[nonzero_weights_cnt > max_nonzeros])
    # any larger than pen_max will give us less non_zeros than required
    l1_pen_max = min(l1_pen_vec[nonzero_weights_cnt < max_nonzeros])

print(l1_pen_min, l1_pen_max)
# min, max = 127, 264 (rounded to nearest int)



l1_pen_vec_2 = np.linspace(l1_pen_min,l1_pen_max,20)
nonzero_weights_cnt_2 = np.zeros(len(l1_pen_vec_2))
rss_vec = np.zeros(len(l1_pen_vec_2))

for i,p in enumerate(l1_pen_vec_2):
    model = linear_model.Lasso(alpha = p, normalize= True)
    model.fit(training[features_to_use], training['price'])
    nonzero_weights_cnt_2[i] = np.count_nonzero(model.coef_) + np.count_nonzero(model.intercept_)
    errors = model.predict(validation[features_to_use]) - validation['price']
    rss_vec[i] = np.sum(errors ** 2)

rss_choice = min(rss_vec[nonzero_weights_cnt_2 == max_nonzeros])
l1_pen_choice = l1_pen_vec_2[rss_vec == rss_choice]
print(l1_pen_choice)
# l1_pen = 156


model = linear_model.Lasso(alpha = l1_pen_choice, normalize = True)
model.fit(training[features_to_use], training['price'])
print_coef(model, features_to_use)






