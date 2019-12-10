# -*- coding: utf-8 -*-
"""
This model takes the winemag data & filters on the north_america and europe continents.
After cleaning and prepping the data, 
we run a bayesian linear models on the data using variational inference & pymc3 
"""

import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
from sklearn.model_selection import train_test_split
import theano
import json
from sklearn.metrics import classification_report

# Use Feature Engineering file to clean data beforehand
from FE import FE

wine = pd.read_csv('winemag-data-130k-v2.csv')
wine_clean = FE(wine).copy().query('continent in ["north_america","europe"]')

# For reproducibility
SEED = 20200514 

# %% Data cleaning/prep
cont_vars = ['price']
cat_vars = ['province','year','color']
pred = 'points'
data = wine_clean.copy()

# Initial Bayesian Linear Regression
colList = cont_vars + cat_vars + [pred]

# Drop where we don't have region data
regression = data.query('country.notnull() & price.notnull()' , engine='python')[colList]

# Break out categorical & one-hot encode
regression_categorical = pd.get_dummies(regression[cat_vars])

# Scale Price Logarithmically & scale
X_mean = np.log(regression[cont_vars]).mean()
X_std = np.log(regression[cont_vars]).std()
regression_continuous = ((np.log(regression[cont_vars]) - X_mean)/X_std)

X = pd.concat([regression_categorical, regression_continuous], axis=1)

# Scale Y
Y_mean = regression[pred].mean()
Y_std = regression[pred].std()
Y = (regression[pred] - Y_mean)/Y_std

# Split out a train/test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=SEED, test_size = 0.20)

train_cat = X_train.drop(cont_vars, axis=1)

# Set up tensors
y_tensor = theano.shared(y_train.values.astype('float64'))
x_cont_tensor = theano.shared(X_train[cont_vars].values.astype('float64'))
x_cat_tensor = theano.shared(X_train.drop(cont_vars, axis=1).values.astype('float64'))

# %% Run bayesian linear regression model 
with pm.Model() as model: 
    zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
    zbetaj_cont = pm.Normal('zbetaj_cont', mu=0, sd=2)
    zbetaj_cat = pm.Beta('zbetaj_cat', alpha = 1, beta = 1, shape=train_cat.shape[1])
    
    zmu =  zbeta0 + pm.math.dot(zbetaj_cont, x_cont_tensor.T) + pm.math.dot(zbetaj_cat, x_cat_tensor.T)
        
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    
    likelihood = pm.Normal('likelihood', mu=zmu, sigma=sigma, observed=y_tensor)

with model:
     advi_fit = pm.fit(n=15000, method = pm.ADVI())

# %% plot the ELBO over time

advi_elbo = pd.DataFrame(
    {'ELBO': -advi_fit.hist,
     'n': np.arange(advi_fit.hist.shape[0])})
    
_ = sns.lineplot(y='ELBO', x='n', data=advi_elbo)

# %% Use the model to make predictions on the test set

# Sample to look at regression fit
advi_trace = advi_fit.sample(10000)

# Switch tensor to test data
y_tensor.set_value(y_test)
x_cont_tensor.set_value(X_test[cont_vars].values)
x_cat_tensor.set_value(X_test.drop(cont_vars, axis=1).values)

# Generate predictions for test set
advi_posterior_pred = pm.sample_posterior_predictive(advi_trace, 1000, model)
advi_predictions = np.mean(advi_posterior_pred['likelihood'], axis=0)

# %% Take the prediction data, scale them to the original scale, and run results
advi_predictions_orig = advi_predictions*Y_std + Y_mean
y_test_orig = y_test*Y_std + Y_mean

prediction_data = pd.DataFrame(
    {'ADVI': advi_predictions_orig[0], 
     'actual': y_test_orig,
     'error_ADVI': advi_predictions_orig[0] - y_test_orig})

# Calculate RMSE
RMSE = np.sqrt(np.mean(prediction_data.error_ADVI ** 2))
print(f'RMSE for ADVI predictions = {RMSE:.3f}')

_ = sns.lmplot(y='ADVI', x='actual', data=prediction_data, 
               line_kws={'color': 'red', 'alpha': 0.5})

class_grid = prediction_data.copy()
class_grid['cat_actual'] = np.where(class_grid['actual'] < 86, "BottomTier",
                                      np.where(class_grid['actual'] < 93, "MiddleTier","TopTier"))
class_grid['cat_predicted'] = np.where(class_grid['ADVI'] < 86, "BottomTier",
                                      np.where(class_grid['ADVI'] < 93, "MiddleTier","TopTier"))

print(classification_report(class_grid['cat_actual'],class_grid['cat_predicted'],target_names=['TopTier','MiddleTier','BottomTier']))

pm.model_to_graphviz(model)

# %%
