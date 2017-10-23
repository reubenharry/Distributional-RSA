import pandas as pd
import numpy as np
import scipy as sp
import statsmodels.api as sm
import statsmodels.formula.api as smf
# import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
np.set_printoptions(threshold='nan') #to print the whole array



xx = np.array([-30.0, 110.0])
yy = np.array([0., 1000.0])
means = [xx.mean(), yy.mean()]  
stds = [xx.std() / 10, yy.std() / 10]
corr = 0.7         # correlation
covs = [[stds[0]**2          , stds[0]*stds[1]*corr],
        [stds[0]*stds[1]*corr,           stds[1]**2]]

m = np.random.multivariate_normal(means, covs, 2500).T
# plt.scatter(m[0], m[1])

# categorical variable
cities = np.random.randint(0, 50, 2500,dtype='i')

# binary variable
kids =  np.random.randint(0, 2, 2500,dtype='i')

df = pd.DataFrame({'ice_cream':m[1], 'temp': m[0], 'cities': cities, 'kids': kids})

# Assuming different cities have different preferences
for num in range(50):
    df['ice_cream'][df['cities']==num] += df['temp']*num

# Assuming liking ice cream is increased with kids (1; no kids = 0)
df['ice_cream'][df['kids']==1] += df['temp']*1.5

# scale
df['ice_cream'] = df['ice_cream']/35

model1 = sm.MixedLM.from_formula("ice_cream ~ temp", data=df, groups=df["cities"]).fit(reml=False)
print(model1.summary())