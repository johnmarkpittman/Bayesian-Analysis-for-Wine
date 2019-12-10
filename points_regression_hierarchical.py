# -*- coding: utf-8 -*-
"""
This model takes the winemag data & filters on the north_america and europe continents.
After cleaning and prepping the date, 
we run two bayesian hierarchical models (one for each continent) on the data using variational inference & pymc3
"""
import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
from sklearn.model_selection import train_test_split
import theano
from sklearn.metrics import classification_report

# Use Feature Engineering file to clean data beforehand
from FE import FE

wine = pd.read_csv('winemag-data-130k-v2.csv')
wine_clean = FE(wine)

# For reproducibility
SEED = 20200514 

# %% Data cleaning/prep

# Clone initial data sets for cleaning
NA = wine_clean.copy().query('continent == "north_america"')
EU = wine_clean.copy().query('continent == "europe"')

# Convert province to codes to be used in hierarchical priors
province_dict = wine_clean.set_index('province').to_dict()['continent']

NA_dict = np.asarray([i for i,j in province_dict.items() if j == "north_america"])
NA_lookup = dict(zip(NA_dict, range(len(NA_dict))))
NA['province'] = NA.province.replace(NA_lookup).values

EU_dict = np.asarray([i for i,j in province_dict.items() if j == "europe"])
EU_lookup = dict(zip(EU_dict, range(len(EU_dict))))
EU['province'] = EU.province.replace(EU_lookup).values

# Set Other & Nan Values to 0
NA_province = NA['province'] = NA['province'].replace(to_replace={np.nan :0}).astype(int).values
EU_province = NA['province'] = NA['province'].replace(to_replace={np.nan :0}).astype(int).values

NA_unique = NA['province'].unique()
EU_unique = EU['province'].unique()

cont_vars = ['price']
cat_vars = ['year','color']
hierarchical = 'province'
pred = 'points'

NA_data = NA.copy()
EU_data = EU.copy()

# List of variables to be used in regression
colList = cont_vars + cat_vars + [hierarchical] + [pred]

# Drop where we don't have region data
regression_NA = NA_data.query('country.notnull() & price.notnull()' , engine='python')[colList]
regression_EU = EU_data.query('country.notnull() & price.notnull()' , engine='python')[colList]

# Break out categorical & one-hot encode
NA_regression_categorical = pd.get_dummies(regression_NA[cat_vars])
EU_regression_categorical = pd.get_dummies(regression_EU[cat_vars])

# Scale Price Logarithmically & scale
NA_X_mean = np.log(regression_NA[cont_vars]).mean()
NA_X_std = np.log(regression_NA[cont_vars]).std()
NA_regression_continuous = ((np.log(regression_NA[cont_vars]) - NA_X_mean)/NA_X_std)

EU_X_mean = np.log(regression_EU[cont_vars]).mean()
EU_X_std = np.log(regression_EU[cont_vars]).std()
EU_regression_continuous = ((np.log(regression_EU[cont_vars]) - EU_X_mean)/EU_X_std)

NA_X = pd.concat([NA_regression_categorical, NA_regression_continuous, regression_NA[hierarchical]], axis=1)
EU_X = pd.concat([EU_regression_categorical, EU_regression_continuous, regression_EU[hierarchical]], axis=1)

# Scale Y
NA_Y_mean = regression_NA[pred].mean()
NA_Y_std = regression_NA[pred].std()
NA_Y = (regression_NA[pred] - NA_Y_mean)/NA_Y_std

# Scale Y
EU_Y_mean = regression_EU[pred].mean()
EU_Y_std = regression_EU[pred].std()
EU_Y = (regression_EU[pred] - EU_Y_mean)/EU_Y_std

# Split out a train/test set
NA_X_train, NA_X_test, NA_y_train, NA_y_test = train_test_split(NA_X, NA_Y, random_state=SEED, test_size = 0.20)
EU_X_train, EU_X_test, EU_y_train, EU_y_test = train_test_split(EU_X, EU_Y, random_state=SEED, test_size = 0.20)

NA_train_cat = NA_X_train.drop(cont_vars + [hierarchical], axis=1)
EU_train_cat = EU_X_train.drop(cont_vars + [hierarchical], axis=1)

# Set up shape of continents priors
NA_len = len(NA_unique)+1
EU_len = len(EU_unique)+1

# Set up tensors for train/test data
NA_y_tensor = theano.shared(NA_y_train.values.astype('float64'))
NA_x_cont_tensor = theano.shared(NA_X_train[cont_vars].values.astype('float64'))
NA_x_cat_tensor = theano.shared(NA_X_train.drop(cont_vars + [hierarchical], axis=1).values.astype('float64'))
NA_x_province_tensor = theano.shared(NA_X_train['province'].values.astype('int64'))

EU_y_tensor = theano.shared(EU_y_train.values.astype('float64'))
EU_x_cont_tensor = theano.shared(EU_X_train[cont_vars].values.astype('float64'))
EU_x_cat_tensor = theano.shared(EU_X_train.drop(cont_vars + [hierarchical], axis=1).values.astype('float64'))
EU_x_province_tensor = theano.shared(EU_X_train['province'].values.astype('int64'))

# %% Bayesian Linear Regression models
with pm.Model() as NA_model: 
    # Hyperpriors
    mu_a = pm.Normal('mu_a',mu=0,sd=1e5)
    sigma_a = pm.HalfCauchy('sigma_a',5)
    
    # Hierarcical intercept
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=NA_len)
    
    # Regular intercept
    zbetaj_cont = pm.Normal('zbetaj_cont', mu=0, sd=2)
    zbetaj_cat = pm.Beta('zbetaj_cat', alpha = 1, beta = 1, shape=NA_train_cat.shape[1])

    zmu =  a[NA_x_province_tensor] + pm.math.dot(zbetaj_cont, NA_x_cont_tensor.T) + pm.math.dot(zbetaj_cat, NA_x_cat_tensor.T)
        
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    
    likelihood = pm.Normal('likelihood', mu=zmu, sigma=sigma, observed=NA_y_tensor)
    
with pm.Model() as EU_model: 
    # Hyperpriors
    mu_a = pm.Normal('mu_a',mu=0,sd=1e5)
    sigma_a = pm.HalfCauchy('sigma_a',5)
    
    # Hierarcical intercept
    a = pm.Normal('a', mu=mu_a, sd=sigma_a, shape=EU_len)
    
    # Regular intercept
    zbetaj_cont = pm.Normal('zbetaj_cont', mu=0, sd=2)
    zbetaj_cat = pm.Beta('zbetaj_cat', alpha = 1, beta = 1, shape=EU_train_cat.shape[1])

    zmu =  a[EU_x_province_tensor] + pm.math.dot(zbetaj_cont, EU_x_cont_tensor.T) + pm.math.dot(zbetaj_cat, EU_x_cat_tensor.T)
        
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    
    likelihood = pm.Normal('likelihood', mu=zmu, sigma=sigma, observed=EU_y_tensor)

with NA_model:
     NA_advi_fit = pm.fit(n=15000, method = pm.ADVI())
     
with EU_model:
     EU_advi_fit = pm.fit(n=15000, method = pm.ADVI())
# %% Plot Elbo Results

# plot the ELBO over time

NA_advi_elbo = pd.DataFrame(
    {'ELBO': -NA_advi_fit.hist,
     'n': np.arange(NA_advi_fit.hist.shape[0])})
    
_ = sns.lineplot(y='ELBO', x='n', data=NA_advi_elbo)
    
EU_advi_elbo = pd.DataFrame(
    {'ELBO': -EU_advi_fit.hist,
     'n': np.arange(EU_advi_fit.hist.shape[0])})
    
_ = sns.lineplot(y='ELBO', x='n', data=EU_advi_elbo)

# %% Run data on test split

# Sample to look at regression fit
NA_advi_trace = NA_advi_fit.sample(10000)
EU_advi_trace = EU_advi_fit.sample(10000)

# Switch tensor to test data
NA_y_tensor.set_value(NA_y_test)
NA_x_cont_tensor.set_value(NA_X_test[cont_vars].values)
NA_x_cat_tensor.set_value(NA_X_test.drop(cont_vars + [hierarchical], axis=1).values)
NA_x_province_tensor.set_value(NA_X_test['province'].values)

EU_y_tensor.set_value(EU_y_test)
EU_x_cont_tensor.set_value(EU_X_test[cont_vars].values)
EU_x_cat_tensor.set_value(EU_X_test.drop(cont_vars + [hierarchical], axis=1).values)
EU_x_province_tensor.set_value(EU_X_test['province'].values)

# Generate predictions for test set
NA_advi_posterior_pred = pm.sample_posterior_predictive(NA_advi_trace, 1000, NA_model)
NA_advi_predictions = np.mean(NA_advi_posterior_pred['likelihood'], axis=0)

EU_advi_posterior_pred = pm.sample_posterior_predictive(EU_advi_trace, 1000, EU_model)
EU_advi_predictions = np.mean(EU_advi_posterior_pred['likelihood'], axis=0)

# %% Convert back to original scale & plot results
NA_advi_predictions_orig = NA_advi_predictions*NA_Y_std + NA_Y_mean
NA_y_test_orig = NA_y_test*NA_Y_std + NA_Y_mean

EU_advi_predictions_orig = EU_advi_predictions*EU_Y_std + EU_Y_mean
EU_y_test_orig = EU_y_test*EU_Y_std + EU_Y_mean

NA_prediction_data = pd.DataFrame(
    {'ADVI': NA_advi_predictions_orig[0], 
     'actual': NA_y_test_orig,
     'error_ADVI': NA_advi_predictions_orig[0] - NA_y_test_orig})
    
EU_prediction_data = pd.DataFrame(
    {'ADVI': EU_advi_predictions_orig[0], 
     'actual': EU_y_test_orig,
     'error_ADVI': EU_advi_predictions_orig[0] - EU_y_test_orig})
    
final_prediction_data = pd.concat([NA_prediction_data, EU_prediction_data], axis=0)

# Calculate RMSE
RMSE = np.sqrt(np.mean(final_prediction_data.error_ADVI ** 2))
print(f'RMSE for ADVI predictions = {RMSE:.3f}')

_ = sns.lmplot(y='ADVI', x='actual', data=final_prediction_data, 
               line_kws={'color': 'red', 'alpha': 0.5})

class_grid = final_prediction_data.copy()
class_grid['cat_actual'] = np.where(class_grid['actual'] < 86, "BottomTier",
                                      np.where(class_grid['actual'] < 93, "MiddleTier","TopTier"))
class_grid['cat_predicted'] = np.where(class_grid['ADVI'] < 86, "BottomTier",
                                      np.where(class_grid['ADVI'] < 93, "MiddleTier","TopTier"))

print(classification_report(class_grid['cat_actual'],class_grid['cat_predicted'],target_names=['TopTier','MiddleTier','BottomTier']))

pm.model_to_graphviz(NA_model)

# %%
