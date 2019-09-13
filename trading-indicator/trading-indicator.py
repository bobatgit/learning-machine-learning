# -*- coding: utf-8 -*-

# Approach: 
# Build an ML-derived price movement / technical indicator:
# 1. Gather a selection of instruments - several un/less correlated ETFs.
# 2. Train model on all instruments, using test sets.
# 3. Constraints: attempt to approximate what MACD and similar technical
#    indicators provide. Limit training data to a fixed ammount of days:
#    - Daily returns only.
#    - Limit training data window to 26 days (same as MACD window).
#    - Forecast 1 day ahead.
#    - Shuffle training datasets to include diverse ETFs.
#    - Minimum 5 years of history. Prefer after 2013.
# 4. Use Yahoo Adj. Close for returns (includes dividends)
#    Ignore pandas_datareader as it's deprecated. Use instead
#    https://aroussi.com/post/python-yahoo-finance
#
# To-do:
# - Replace Yahoo with more reliable source (eg. Interactive Brokers). 
#   Make sure dividends are adequately calculated in the returns.
# - Include fixed income or other ETFs.
# - Forecase up to 3 and 5 days in advance (build 2x extra models).
# - Scale forecasts to a fixed range of -10.0 and +10.0, 
#   and limit their precision to 1 decimal point max.
# - Extend to 10 or 15 years of historic data.
#
# Define the instruments to use:
# Selecting global ETFs which have seen both good & bad macro economic conditions.
# Adjust as needed for your own models.
# - US equity: SPY, DIA
# - UK equity: FTSE250 (0P0000WN7D.L)
# - Europe equity: EZU
# - China equity: FXI
# - Japan equity: EWJ - note: 2006/01/11 returns need to be flattened
# - Brazil equity: EWZ
# - Russia equity: RSX - note: data available only for after 2008


import pandas as pd
import yfinance as yf
import numpy as np
import os
from datetime import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import style

# Define the instrument tickers
instrument_list = ['SPY', 'DIA', '0P0000WN7D.L', 'EZU', 'FXI', 'EWJ', 'EWZ', 'RSX']
# Define the start and end date in YYYY-MM-DD
start = '2013-01-01'
end = '2019-08-31' # This is good enough
days = 26
test_data_fract = 0.05


# Download the instruments only once. Conserve bandwidth
def load_data(instrument_list, start, end):
    files = []
    print(' ---- Loading data ---- ')
    for i in instrument_list:
        i_name = ''.join([i, '.csv'])
        if not os.path.exists(i_name):
            print(' -- Downloading', i, ' -- ')
            data = yf.download(i, start=start, end=end, auto_adjust=True)
            data.to_csv(i_name)
        data = pd.read_csv(i_name, index_col=0)
        files.append(data)
    return pd.concat(files, 
                     keys=instrument_list, 
                     axis=1, 
                     names=['instrument','value'], 
                     sort=False)


# Open the daily prices
prices = load_data(instrument_list, start, end)
prices = prices.dropna()
# Add some quality of life convenience :)
prices.index = pd.to_datetime(prices.index)
prices.columns = prices.columns.swaplevel()
prices = prices.rename(columns={'0P0000WN7D.L':'FTS'})
# Take the closing price for all further calculation
prices = prices.Close


# Calculate daily returns using log approach as it gives better normalisation.
# Note: np.log() is the natural log at base e (not base 10).
rets = np.log(prices).diff(1)
rets = rets.dropna()


# Generate plots to visualise the instruments
# Prices over time
prices.plot()
# Returns over time
rets.plot()
# Scatter plots by pair + normal distribution
pd.plotting.scatter_matrix(rets, diagonal='kde', figsize=(8, 8))
# Correlation by pair
corr = rets.corr()
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)
cax = ax.matshow(corr,cmap='coolwarm', vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(rets.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(rets.columns)
ax.set_yticklabels(rets.columns)
#ax.tick_params(labelsize=2)
plt.show()
# Acerage risk/return by instrument scatter plot
fig = plt.figure()
plt.scatter(rets.mean(), rets.std())
plt.xlabel('Expected returns')
plt.ylabel('Risk')
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (20, -20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


# Define the X and Y datasets for training the ML model
def make_x_y_dataset(rets, days):
    x, y = [], []
    label_days = [''.join(['day_', str(x+1)]) for x in reversed(range(52))]
    for instrument in rets.stack().groupby(level=1):
        print(' -- Making dataset for: ', instrument[0], ' -- ')
        instrument = instrument[1]
        for window in range(len(instrument) - days - 1):
            temp = instrument[window:window+days+1].values
            x.append(pd.DataFrame(dict(zip(label_days, temp[0:days])), index=[0]))
            y.append(pd.DataFrame({'y':temp[-1:]}))
    x = pd.concat(x).reset_index()
    y = pd.concat(y).reset_index()
    return x, y
x, y = make_x_y_dataset(rets, days)


# Define a simple linear model to predict the next day's return
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Supress sklearn warnings
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)


# Define the test and train datasets
# Set aside 5% for testing data
index_test = x.sample(frac=test_data_fract, random_state=42).index
X_train = x.iloc[x.index.difference(index_test)]
y_train = y.iloc[x.index.difference(index_test)]
X_test = x.iloc[index_test]
y_test = y.iloc[index_test]

# Scale the dataset
# Not really needed as the returns data are naturally scaled due to np.log()
# But it's good practice, so might as well do it
from sklearn import preprocessing
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.fit_transform(X_train)
y_train = scaler.fit_transform(y_train)
X_test = scaler.fit_transform(X_test)
y_test = scaler.fit_transform(y_test)


print(' ---- Training models ---- ')
# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)

# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test,y_test)

# KNN Regression
# Try all the values of K for 2 to number of days
confidenceknn = [] #to store rmse values for different k
Ks = range(2, days)
for K in Ks:
    clfknn = KNeighborsRegressor(n_neighbors = K)
    clfknn.fit(X_train, y_train)  #fit the model
    score = clfknn.score(X_test, y_test) #make prediction on test set
    confidenceknn.append(score) #store rmse values


# Results
print(' ---- Results ---- ')
print('Linear regression', confidencereg, '\n',
      'Quadratic regression', confidencepoly2, '\n', 
      'Polynomial regression', confidencepoly3, '\n', 
      'KNN regressor with K=', Ks[confidenceknn.index(max(confidenceknn))], 
      max(confidenceknn))


# Conclusion:
# Seems like the KNN regressor has the best accuracy at K=2, but not my much.
# Note: at days=52 the KNN regressor best accuracy increases to 0.296 with K=3
# Note: at days=104 the KNN regressor best accuracy decreases to 0.218 with K=4
# Note: at days=208 the KNN regressor best accuracy decreases to 0.268 with K=2

# Plot the KNN predicted vs Actual results
# Re-build the KNN model with the best K
clfknn = clfknn = KNeighborsRegressor(n_neighbors = Ks[confidenceknn.index(max(confidenceknn))])
clfknn.fit(X_train, y_train)
X_predicted = clfknn.predict(X_test)
fig = plt.Figure(figsize=(8, 8))
plt.scatter(pd.DataFrame(X_predicted)[1], 
            pd.DataFrame(y_test)[1])
plt.xlabel('Predicted returns')
plt.ylabel('Actual returns')

