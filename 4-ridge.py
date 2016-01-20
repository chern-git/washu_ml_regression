# Week 4 - Ridge regression Assignment 1

import pandas as pd

dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float,
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float,
              'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float,
              'floors':float, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int,
              'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}

sales = pd.read_csv('data/kc_house_data.csv', dtype=dtype_dict)
sales = sales.sort(['sqft_living','price'])
l2_small_penalty = 1.5e-5

import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

sqft_lvg = sales['sqft_living']
sqft_lvg = sqft_lvg.reshape(-1,1)

# poly15_data = polynomial_sframe(sales['sqft_living'], 15) # use equivalent of `polynomial_sframe`
poly15 = PolynomialFeatures(degree=15)
sqft_lvg_poly15 = poly15.fit_transform(sqft_lvg)
sqft_lvg_poly15 = sqft_lvg_poly15.reshape(sqft_lvg_poly15.shape[0],16)

# model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
# model.fit(poly15_data, sales['price'])
model = linear_model.Ridge(alpha=l2_small_penalty, normalize=True)
model.fit(sqft_lvg_poly15, sales['price'])
print(model.coef_[1]) # 124.873306




# Part 2

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

set_1 = pd.read_csv('data/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('data/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('data/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('data/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
l2_small_penalty=1e-9

poly15 = PolynomialFeatures(degree=15)

set1_poly15 = poly15.fit_transform(set_1['sqft_living'].reshape(-1,1))
model_1 = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
model_1.fit(set1_poly15, set_1['price'])
print(model_1.coef_[1]) # 544.669374363

set2_poly15 = poly15.fit_transform(set_2['sqft_living'].reshape(-1,1))
model_2 = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
model_2.fit(set2_poly15, set_2['price'])
print(model_2.coef_[1]) # 859.36262985

set3_poly15 = poly15.fit_transform(set_3['sqft_living'].reshape(-1,1))
model_3 = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
model_3.fit(set3_poly15, set_3['price'])
print(model_3.coef_[1]) # -755.395904775

set4_poly15 = poly15.fit_transform(set_4['sqft_living'].reshape(-1,1))
model_4 = linear_model.Ridge(alpha = l2_small_penalty, normalize = True)
model_4.fit(set4_poly15, set_4['price'])
print(model_4.coef_[1]) # 1119.44565688




# Part 3

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

set_1 = pd.read_csv('data/wk3_kc_house_set_1_data.csv', dtype=dtype_dict)
set_2 = pd.read_csv('data/wk3_kc_house_set_2_data.csv', dtype=dtype_dict)
set_3 = pd.read_csv('data/wk3_kc_house_set_3_data.csv', dtype=dtype_dict)
set_4 = pd.read_csv('data/wk3_kc_house_set_4_data.csv', dtype=dtype_dict)
l2_large_penalty=1.23e2

poly15 = PolynomialFeatures(degree=15)

set1_poly15 = poly15.fit_transform(set_1['sqft_living'].reshape(-1,1))
model_1 = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
model_1.fit(set1_poly15, set_1['price'])
print(model_1.coef_[1]) # 2.32806802958

set2_poly15 = poly15.fit_transform(set_2['sqft_living'].reshape(-1,1))
model_2 = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
model_2.fit(set2_poly15, set_2['price'])
print(model_2.coef_[1]) # 2.09756902778

set3_poly15 = poly15.fit_transform(set_3['sqft_living'].reshape(-1,1))
model_3 = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
model_3.fit(set3_poly15, set_3['price'])
print(model_3.coef_[1]) # 2.28906258119

set4_poly15 = poly15.fit_transform(set_4['sqft_living'].reshape(-1,1))
model_4 = linear_model.Ridge(alpha = l2_large_penalty, normalize = True)
model_4.fit(set4_poly15, set_4['price'])
print(model_4.coef_[1]) # 2.08596194092




# Part 4 - Cross validation - Approach 1

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures

train_valid_shuffled = pd.read_csv('data/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)

def k_fold_cross_validation(k, l2_penalty, data, output):
    for i in range(k):
        start = (n * i) / k
        end = (n * (i + 1)) / k
        data_val = data[start:end+1]
        data_trg = data[0:start].append(data[end+1:n])

        model = linear_model.Ridge(alpha = l2_penalty, normalize = True)
        model.fit(data_trg, data_trg['price'])
        pred_val = model.predict(data_val)

        np.sum((pred_val - data_val['price']) ** 2)

    return avg_val_error


# n = len(train_valid_shuffled)
# k = 10 # 10-fold cross-validation
#
# for i in range(k):
#     start = (n * i) / k
#     end = (n * (i + 1)) / k
#     print(i, (start, end))
    # data_val = data[start:end+1]
    # data_trg = data[0:start].append(data[end+1:n])



# Part 4 - Cross validation - Approach 2

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cross_validation import cross_val_score

train_valid_shuffled = pd.read_csv('data/wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
test = pd.read_csv('data/wk3_kc_house_test_data.csv', dtype=dtype_dict)
l2_set = np.logspace(3, 9, num=13)

poly15 = PolynomialFeatures(degree= 15)
X_train = poly15.fit_transform(train_valid_shuffled['sqft_living'].reshape(-1,1))
X_test = poly15.transform(test['sqft_living'].reshape(-1,1))
y_train = train_valid_shuffled['price']

for i, l2 in enumerate(l2_set):
    model = linear_model.Ridge(alpha = l2, normalize = True)
    scores = cross_val_score(model, X_train, y_train, cv=10)
    print("Using L2 of ", l2, "| Mean score: ", scores.mean())
#  -0.000600028584951


# Training on test set:

model = linear_model.Ridge(alpha =3.16227766e+03, normalize = True)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
sum((y_pred - test['price']) ** 2) # 284682323929148




