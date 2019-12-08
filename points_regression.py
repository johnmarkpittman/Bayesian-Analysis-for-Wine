# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 15:18:28 2019

@author: DanielW20
"""

import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
from sklearn.model_selection import train_test_split
import theano
import json

from FE import FE

wine = pd.read_csv('winemag-data-130k-v2.csv')
wine_clean = FE(wine)

def runModel(data,cont_vars,cat_vars,pred,seed):
    # Initial Bayesian Linear Regression
    colList = cont_vars + cat_vars + [pred]
    
    # Drop where we don't have region data
    regression = data.query('country.notnull() & price.notnull()' , engine='python')[colList]
    
    # Break out categorical & one-hot encode
    regression_categorical = pd.get_dummies(regression[['country','year','continent', 'color']])
    
    # Break out continuous & scale
    X_mean = regression[cont_vars].mean()
    X_std = regression[cont_vars].std()
    regression_continuous = ((regression[cont_vars] - X_mean)/X_std)
    
    X = pd.concat([regression_categorical, regression_continuous], axis=1)
    
    # Scale Y
    Y_mean = regression[pred].mean()
    Y_std = regression[pred].std()
    Y = (regression[pred] - Y_mean)/Y_std
    
    # Split out a train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=SEED, test_size = 0.20)
    
    train_cat = X_train.drop(cont_vars, axis=1)
    
    # Set up tensors & minibatches
    y_tensor = theano.shared(y_train.values.astype('float64'))
    x_cont_tensor = theano.shared(X_train[cont_vars].values.astype('float64'))
    x_cat_tensor = theano.shared(X_train.drop(cont_vars, axis=1).values.astype('float64'))
    
    map_tensor_batch = {y_tensor: pm.Minibatch(y_train.values.astype('float64'), 100),
                        x_cont_tensor: pm.Minibatch(X_train[cont_vars].values.astype('float64'), 100),
                        x_cat_tensor: pm.Minibatch(X_train.drop(cont_vars, axis=1).values.astype('float64'), 100)}
    
    with pm.Model() as model: 
        zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
        zbetaj_cont = pm.Normal('zbetaj_cont', mu=0, sd=2)
        zbetaj_cat = pm.Beta('zbetaj_cat', alpha = 1, beta = 1, shape=train_cat.shape[1])
        
        zmu =  zbeta0 + pm.math.dot(zbetaj_cont, x_cont_tensor.T) + pm.math.dot(zbetaj_cat, x_cat_tensor.T)
            
        nu = pm.Exponential('nu', 1/29.)
        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
        
        likelihood = pm.StudentT('likelihood', nu=nu, mu=zmu, sigma=sigma, observed=y_tensor)
    
    with model:
         advi_fit = pm.fit(n=15000, method = pm.ADVI(), more_replacements=map_tensor_batch)
    
    # plot the ELBO over time
    
    advi_elbo = pd.DataFrame(
        {'ELBO': -advi_fit.hist,
         'n': np.arange(advi_fit.hist.shape[0])})
        
    _ = sns.lineplot(y='ELBO', x='n', data=advi_elbo)
    
    # Sample to look at regression fit
    advi_trace = advi_fit.sample(10000)
    pm.summary(advi_trace)
    
    # Switch tensor to test data
    y_tensor.set_value(y_test)
    x_cont_tensor.set_value(X_test[cont_vars].values)
    x_cat_tensor.set_value(X_test.drop(cont_vars, axis=1).values)
    
    # Generate predictions for test set
    advi_posterior_pred = pm.sample_posterior_predictive(advi_trace, 1000, model)
    advi_predictions = np.mean(advi_posterior_pred['likelihood'], axis=0)
    
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
    
# For reproducibility
SEED = 20200514 

cont_vars = ['price']
cat_vars = ['country','year','continent', 'color']
pred = 'points'

runModel(wine_clean, cont_vars, cat_vars,pred,SEED)

def runModelHierarchical(data,cont_vars,cat_vars,pred,seed):
    # Initial Bayesian Linear Regression
    colList = cont_vars + cat_vars + [pred]
    
    # Drop where we don't have region data
    regression = data.query('country.notnull() & price.notnull()' , engine='python')[colList]
    
    # Break out categorical & one-hot encode
    regression_categorical = pd.get_dummies(regression[['country','year','continent', 'color']])
    
    # Break out continuous & scale
    X_mean = regression[cont_vars].mean()
    X_std = regression[cont_vars].std()
    regression_continuous = ((regression[cont_vars] - X_mean)/X_std)
    
    X = pd.concat([regression_categorical, regression_continuous], axis=1)
    
    # Scale Y
    Y_mean = regression[pred].mean()
    Y_std = regression[pred].std()
    Y = (regression[pred] - Y_mean)/Y_std
    
    # Split out a train/test set
    X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=SEED, test_size = 0.20)
    
    train_cat = X_train.drop(cont_vars, axis=1)
    
    # Set up tensors & minibatches
    y_tensor = theano.shared(y_train.values.astype('float64'))
    x_cont_tensor = theano.shared(X_train[cont_vars].values.astype('float64'))
    x_cat_tensor = theano.shared(X_train.drop(cont_vars, axis=1).values.astype('float64'))
    x_NA_tensor = theano.shared(X_train['continent_north_america'].values.astype('float64')),
    x_SA_tensor = theano.shared(X_train['continent_south_america'].values.astype('float64')),
    x_AF_tensor = theano.shared(X_train['continent_africa'].values.astype('float64')),
    x_AS_tensor = theano.shared(X_train['continent_asia'].values.astype('float64')),
    x_EU_tensor = theano.shared(X_train['continent_europe'].values.astype('float64')),
    x_OC_tensor = theano.shared(X_train['continent_oceania'].values.astype('float64'))
    
    map_tensor_batch = {y_tensor: pm.Minibatch(y_train.values.astype('float64'), 100),
    x_cont_tensor: pm.Minibatch(X_train[cont_vars].values.astype('float64'), 100),
    x_cat_tensor: pm.Minibatch(X_train.drop(cont_vars, axis=1).values.astype('float64'), 100),
    x_cat_tensor: pm.Minibatch(X_train.drop(cont_vars, axis=1).values.astype('float64'), 100),
    x_NA_tensor: pm.Minibatch(X_train['continent_north_america'].values.astype('float64'), 100),
    x_SA_tensor: pm.Minibatch(X_train['continent_south_america'].values.astype('float64'), 100),
    x_AF_tensor: pm.Minibatch(X_train['continent_africa'].values.astype('float64'), 100),
    x_AS_tensor: pm.Minibatch(X_train['continent_asia'].values.astype('float64'), 100),
    x_EU_tensor: pm.Minibatch(X_train['continent_europe'].values.astype('float64'), 100),
    x_OC_tensor: pm.Minibatch(X_train['continent_oceania'].values.astype('float64'), 100)}
    
    with pm.Model() as model: 
        zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
        zbetaj_cont = pm.Normal('zbetaj_cont', mu=0, sd=2)
        zbetaj_cat = pm.Beta('zbetaj_cat', alpha = 1, beta = 1, shape=train_cat.shape[1])
        
        #Hierarchical Priors
        NA_a = pm.HalfFlat('a')
        NA_b = pm.HalfFlat('b')
        NA_beta = pm.Beta('NA_beta', alpha=NA_a, beta=NA_b, shape=#number of countries)
        
        SA_a = pm.HalfFlat('a')
        SA_b = pm.HalfFlat('b')
        SA_beta = pm.Beta('SA_beta', alpha=NA_a, beta=NA_b)
        
        AF_a = pm.HalfFlat('a')
        AF_b = pm.HalfFlat('b')
        AF_beta = pm.Beta('AF_beta', alpha=NA_a, beta=NA_b)
        
        AS_a = pm.HalfFlat('a')
        AS_b = pm.HalfFlat('b')
        AS_beta = pm.Beta('AS_beta', alpha=NA_a, beta=NA_b)
        
        EU_a = pm.HalfFlat('a')
        EU_b = pm.HalfFlat('b')
        EU_beta = pm.Beta('NA_beta', alpha=NA_a, beta=NA_b)
        
        OC_a = pm.HalfFlat('a')
        OC_b = pm.HalfFlat('b')
        OC_beta = pm.Beta('NA_beta', alpha=NA_a, beta=NA_b)
        
        zmu =  zbeta0 + pm.math.dot(zbetaj_cont, x_cont_tensor.T) 
                      + pm.math.dot(zbetaj_cat, x_cat_tensor.T)
                      + pm.math.dot(NA_beta[country], x_NA_tensor,T)
                      + pm.math.dot(SA_beta, x_SA_tensor,T)
                      + pm.math.dot(AF_beta, x_AF_tensor,T)
                      + pm.math.dot(AS_beta, x_AS_tensor,T)
                      + pm.math.dot(EU_beta, x_EU_tensor,T)
                      + pm.math.dot(OC_beta, x_OC_tensor,T)
            
        nu = pm.Exponential('nu', 1/29.)
        sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
        
        likelihood = pm.StudentT('likelihood', nu=nu, mu=zmu, sigma=sigma, observed=y_tensor)
    
    with model:
         advi_fit = pm.fit(n=15000, method = pm.ADVI(), more_replacements=map_tensor_batch)
    
    # plot the ELBO over time
    
    advi_elbo = pd.DataFrame(
        {'ELBO': -advi_fit.hist,
         'n': np.arange(advi_fit.hist.shape[0])})
        
    _ = sns.lineplot(y='ELBO', x='n', data=advi_elbo)
    
    # Sample to look at regression fit
    advi_trace = advi_fit.sample(10000)
    pm.summary(advi_trace)
    
    # Switch tensor to test data
    y_tensor.set_value(y_test)
    x_cont_tensor.set_value(X_test['price'].values)
    x_cat_tensor.set_value(X_test.drop('price', axis=1).values)
    
    # Generate predictions for test set
    advi_posterior_pred = pm.sample_posterior_predictive(advi_trace, 1000, model)
    advi_predictions = np.mean(advi_posterior_pred['likelihood'], axis=0)
    
    advi_predictions_orig = advi_predictions*Y_std + Y_mean
    y_test_orig = y_test*Y_std + Y_mean
    
    prediction_data = pd.DataFrame(
        {'ADVI': advi_predictions_orig, 
         'actual': y_test_orig,
         'error_ADVI': advi_predictions_orig - y_test_orig})
    
    
    # Calculate RMSE
    RMSE = np.sqrt(np.mean(prediction_data.error_ADVI ** 2))
    print(f'RMSE for ADVI predictions = {RMSE:.3f}')
    
    _ = sns.lmplot(y='ADVI', x='actual', data=prediction_data, 
                   line_kws={'color': 'red', 'alpha': 0.5})
    
# Gather dictionary data
with open('country_dict.json', 'r') as fp:
    country_dict = json.load(fp)

NA = np.asarray([i for i,j in country_dict.items() if j == "north_america"])
NA_lookup = dict(zip(NA, range(len(NA))))
wine_clean['country'] = wine_clean.country.replace(NA_lookup).values
