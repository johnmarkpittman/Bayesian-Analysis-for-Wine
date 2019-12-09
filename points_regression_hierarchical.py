# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:20:36 2019

@author: DanielW20
"""
import pandas as pd
import numpy as np
import pymc3 as pm
import seaborn as sns
from sklearn.model_selection import train_test_split
import theano

from FE import FE

wine = pd.read_csv('winemag-data-130k-v2.csv')
wine_clean = FE(wine)

# For reproducibility
SEED = 20200514 

wine_hierarchy = wine_clean.copy()

# Gather dictionary data & encode for use in priors
country_dict = wine_clean.set_index('country').to_dict()['continent']

NA = np.asarray([i for i,j in country_dict.items() if j == "north_america"])
NA_lookup = dict(zip(NA, range(len(NA))))
wine_hierarchy['country'] = wine_hierarchy.country.replace(NA_lookup).values

SA = np.asarray([i for i,j in country_dict.items() if j == "south_america"])
SA_lookup = dict(zip(SA, range(len(SA))))
wine_hierarchy['country'] = wine_hierarchy.country.replace(SA_lookup).values

AS = np.asarray([i for i,j in country_dict.items() if j == "asia"])
AS_lookup = dict(zip(AS, range(len(AS))))
wine_hierarchy['country'] = wine_hierarchy.country.replace(AS_lookup).values

AF = np.asarray([i for i,j in country_dict.items() if j == "africa"])
AF_lookup = dict(zip(AF, range(len(AF))))
wine_hierarchy['country'] = wine_hierarchy.country.replace(AF_lookup).values

EU = np.asarray([i for i,j in country_dict.items() if j == "europe"])
EU_lookup = dict(zip(EU, range(len(EU))))
wine_hierarchy['country'] = wine_hierarchy.country.replace(EU_lookup).values

OC = np.asarray([i for i,j in country_dict.items() if j == "oceania"])
OC_lookup = dict(zip(OC, range(len(OC))))
wine_hierarchy['country'] = wine_hierarchy.country.replace(OC_lookup).values

# Set Other & Nan Values to 0
country = wine_hierarchy['country'] = wine_hierarchy['country'].replace(to_replace={np.nan :0}).astype(int).values

NA_unique = wine_hierarchy.query('continent == "north_america"')['country'].unique()
SA_unique = wine_hierarchy.query('continent == "south_america"')['country'].unique()
AF_unique = wine_hierarchy.query('continent == "africa"')['country'].unique()
AS_unique = wine_hierarchy.query('continent == "asia"')['country'].unique()
EU_unique = wine_hierarchy.query('continent == "europe"')['country'].unique()
OC_unique = wine_hierarchy.query('continent == "oceania"')['country'].unique()

cont_vars = ['price']
cat_vars = ['year','continent', 'color']
hierarchical = 'country'
pred = 'points'

data = wine_hierarchy.copy()

# Initial Bayesian Linear Regression
colList = cont_vars + cat_vars + [hierarchical] + [pred]

# Drop where we don't have region data
regression = data.query('country.notnull() & price.notnull()' , engine='python')[colList]

# Break out categorical & one-hot encode
regression_categorical = pd.get_dummies(regression[cat_vars])

# Scale Price Logarithmically & scale
X_mean = np.log(regression[cont_vars]).mean()
X_std = np.log(regression[cont_vars]).std()
regression_continuous = ((np.log(regression[cont_vars]) - X_mean)/X_std)

X = pd.concat([regression_categorical, regression_continuous, regression[hierarchical]], axis=1)

# Scale Y
Y_mean = regression[pred].mean()
Y_std = regression[pred].std()
Y = (regression[pred] - Y_mean)/Y_std

# Create list of filters for hierarchical model
def continentFilter(country, countries):
    if country in countries:
        return country
    else:
        return 0

# Create Indices
NA_country = pd.DataFrame.from_dict(dict(map(lambda i,j: (j,continentFilter(i,NA_unique)), X['country'],X.index)), orient='index', columns=["NA_country"])
SA_country = pd.DataFrame.from_dict(dict(map(lambda i,j: (j,continentFilter(i,SA_unique)), X['country'],X.index)), orient='index', columns=["SA_country"])
AF_country = pd.DataFrame.from_dict(dict(map(lambda i,j: (j,continentFilter(i,AF_unique)), X['country'],X.index)), orient='index', columns=["AF_country"])
AS_country = pd.DataFrame.from_dict(dict(map(lambda i,j: (j,continentFilter(i,AS_unique)), X['country'],X.index)), orient='index', columns=["AS_country"])
EU_country = pd.DataFrame.from_dict(dict(map(lambda i,j: (j,continentFilter(i,EU_unique)), X['country'],X.index)), orient='index', columns=["EU_country"])
OC_country = pd.DataFrame.from_dict(dict(map(lambda i,j: (j,continentFilter(i,OC_unique)), X['country'],X.index)), orient='index', columns=["OC_country"])

# Merge indices into dataset
X = pd.concat([X, NA_country,SA_country,AF_country,AS_country,EU_country,OC_country], axis=1)

# Split out a train/test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=SEED, test_size = 0.20)

continent = ['continent_north_america','continent_south_america', 'continent_africa', 'continent_asia', 'continent_europe', 'continent_oceania']
train_cat = X_train.drop(cont_vars + [hierarchical] + continent, axis=1)

# Set up shape of continents priors
NA = len(NA_unique)+1
SA = len(SA_unique)+1
AS = len(AS_unique)+1
AF = len(AF_unique)+1
EU = len(EU_unique)+1
OC = len(OC_unique)+1

# Set up tensors
y_tensor = theano.shared(y_train.values.astype('float64'))
x_cont_tensor = theano.shared(X_train[cont_vars].values.astype('float64'))
x_cat_tensor = theano.shared(X_train.drop(cont_vars + [hierarchical] + continent, axis=1).values.astype('float64'))
x_country_tensor = theano.shared(X_train['country'].values.astype('float64'))
x_NA_tensor = theano.shared(X_train['continent_north_america'].values.astype('float64'))
x_SA_tensor = theano.shared(X_train['continent_south_america'].values.astype('float64'))
x_AF_tensor = theano.shared(X_train['continent_africa'].values.astype('float64'))
x_AS_tensor = theano.shared(X_train['continent_asia'].values.astype('float64'))
x_EU_tensor = theano.shared(X_train['continent_europe'].values.astype('float64'))
x_OC_tensor = theano.shared(X_train['continent_oceania'].values.astype('float64'))
index_NA_tensor = theano.shared(X_train['NA_country'].values.astype('int64'))
index_SA_tensor = theano.shared(X_train['SA_country'].values.astype('int64'))
index_AF_tensor = theano.shared(X_train['AF_country'].values.astype('int64'))
index_AS_tensor = theano.shared(X_train['AS_country'].values.astype('int64'))
index_EU_tensor = theano.shared(X_train['EU_country'].values.astype('int64'))
index_OC_tensor = theano.shared(X_train['OC_country'].values.astype('int64'))

with pm.Model() as model: 
    zbeta0 = pm.Normal('zbeta0', mu=0, sd=2)
    zbetaj_cont = pm.Normal('zbetaj_cont', mu=0, sd=2)
    zbetaj_cat = pm.Beta('zbetaj_cat', alpha = 1, beta = 1, shape=train_cat.shape[1])
    
    #Hierarchical Priors
    NA_a = pm.HalfFlat('NA_a')
    NA_b = pm.HalfFlat('NA_b')
    NA_beta = pm.Beta('NA_beta', alpha=NA_a, beta=NA_b, shape=NA)
    
    SA_a = pm.HalfFlat('SA_a')
    SA_b = pm.HalfFlat('SA_b')
    SA_beta = pm.Beta('SA_beta', alpha=SA_a, beta=SA_b, shape=SA)
    
    AF_a = pm.HalfFlat('AF_a')
    AF_b = pm.HalfFlat('AF_b')
    AF_beta = pm.Beta('AF_beta', alpha=AF_a, beta=AF_b, shape=AF)
    
    AS_a = pm.HalfFlat('AS_a')
    AS_b = pm.HalfFlat('AS_b')
    AS_beta = pm.Beta('AS_beta', alpha=AS_a, beta=AS_b, shape=AS)
    
    EU_a = pm.HalfFlat('EU_a')
    EU_b = pm.HalfFlat('EU_b')
    EU_beta = pm.Beta('EU_beta', alpha=EU_a, beta=EU_b, shape=EU)
    
    OC_a = pm.HalfFlat('OC_a')
    OC_b = pm.HalfFlat('OC_b')
    OC_beta = pm.Beta('OC_beta', alpha=OC_a, beta=OC_b, shape=OC)
    
    zmu =  zbeta0 + pm.math.dot(zbetaj_cont, x_cont_tensor.T) + \
                    pm.math.dot(zbetaj_cat, x_cat_tensor.T) + \
                    pm.math.dot(NA_beta[index_NA_tensor], x_NA_tensor.T) + \
                    pm.math.dot(SA_beta[index_SA_tensor], x_SA_tensor.T) + \
                    pm.math.dot(AF_beta[index_AF_tensor], x_AF_tensor.T) + \
                    pm.math.dot(AS_beta[index_AS_tensor], x_AS_tensor.T) + \
                    pm.math.dot(EU_beta[index_EU_tensor], x_EU_tensor.T) + \
                    pm.math.dot(OC_beta[index_OC_tensor], x_OC_tensor.T)
        
    sigma = pm.HalfCauchy('sigma', beta=10, testval=1.)
    
    likelihood = pm.Normal('likelihood', mu=zmu, sigma=sigma, observed=y_tensor)

with model:
     advi_fit = pm.fit(n=15000, method = pm.ADVI())

# plot the ELBO over time

advi_elbo = pd.DataFrame(
    {'ELBO': -advi_fit.hist,
     'n': np.arange(advi_fit.hist.shape[0])})
    
_ = sns.lineplot(y='ELBO', x='n', data=advi_elbo)

# Sample to look at regression fit
advi_trace = advi_fit.sample(10000)

# Switch tensor to test data
y_tensor.set_value(y_test)
x_cont_tensor.set_value(X_test[cont_vars].values)
x_cat_tensor.set_value(X_test.drop(cont_vars + [hierarchical]+continent, axis=1).values)
x_country_tensor.set_value(X_test['country'].values)
x_NA_tensor.set_value(X_test['continent_north_america'].values)
x_SA_tensor.set_value(X_test['continent_south_america'].values)
x_AF_tensor.set_value(X_test['continent_africa'].values)
x_AS_tensor.set_value(X_test['continent_asia'].values)
x_EU_tensor.set_value(X_test['continent_europe'].values)
x_OC_tensor.set_value(X_test['continent_oceania'].values)

index_NA_tensor.set_value(X_test['NA_country'].values)
index_SA_tensor.set_value(X_test['SA_country'].values)
index_AF_tensor.set_value(X_test['AF_country'].values)
index_AS_tensor.set_value(X_test['AS_country'].values)
index_EU_tensor.set_value(X_test['EU_country'].values)
index_OC_tensor.set_value(X_test['OC_country'].values)

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