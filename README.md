# Wine Enthusiast dataset - 
## Classification models using LDA, Naive Bayes, & text mining 
## Bayesian Linear Regression models to predict points assigned by wine reviewer

FE.py is based on the following git page: https://github.com/gorokhovnik/wine_analysis/

Main files include: 
1. **points_regression.py** - this runs a bayesian regression model on the wine data set using ADVI
2. **points_regression_hierarchical.py** - this runs a bayesian hierarchical regression model on the wine data set using ADVI
    - In this model we use hyperpriors for each continent on province to create the multi-level model
3. **text_mining_with_naive_bayes.ipynb** - This creates a dataframe from the description of the wine dataset using text mining
    - Then uses that dataframe to run naive bayes classification models
4. **Vino_LDA.ipynb** - This produces an LDA model from description & uses the topic proportions to train classification models
