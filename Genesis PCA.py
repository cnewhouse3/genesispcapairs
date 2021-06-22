from mlpairs import OpticsPairs
from mlfinlab.optimal_mean_reversion import OrnsteinUhlenbeck
import pandas as pd
import numpy as np

'''Taking in dataframe with time-series price data for each crypto, making sure there's no nans
then initalizing train/test data'''

stock_prices = pd.read_csv('morepairs.csv',index_col=0,parse_dates=True)
stock_prices = stock_prices.replace(np.nan,0)

train = stock_prices[:'2021']
test = stock_prices['2021':]

train.head()

op = OpticsPairs(train)
op.returns.head()

op.reduce_PCA()
op.plot_loadings()

op.plot_explained_variance()
total_variance_explained = np.cumsum(op.explained_variance_ratio_)[-1]
print(f"Total variance explained: {round(total_variance_explained, 2)*100}%")

'''Running through find_pairs methods and utilizes clustering through the OPTICS algorithm after PCA to find potential pairs,
later on we'll break down criteria to determine'''

op.find_pairs()
op.pairs

op.plot_clusters(n_dimensions=2)

op.calc_eg_norm_spreads()
op.calc_hurst_exponents()
op.calc_half_lives()
op.calc_avg_cross_count()

'''Calling a method to go through data and analyze based on P < 0.05, Hurst  < 0.5, Half-Life of 365 (Trading Days),
How many times the pair crosses back through the mean. If it's 5, we deviate from the mean line and revert back through 5 times
'''
op.filter_pairs()
op.filtered_pairs
print(op.filtered_pairs)
for i in op.filtered_pairs.index:
    op.plot_pair_price_spread(idx=i)
    

#Working on this right now to optimize for cryptos since I don't have enough data to run through train/test
#but this will utilize machine learning to determine the fit of our test data to the model and yield a portfolio optimization
#entry point
'''
for i in range(1):
    trading_pair = list(op.filtered_pairs['pair'].iloc[i])
    
    ou = OrnsteinUhlenbeck()
    
    ou.fit(train[trading_pair],data_frequency='D', discount_rate=[0.01, 0.01], transaction_cost = [0,0], stop_loss=None)
    
    print(f'Model Parameters: {trading_pair}', '\n', ou.check_fit(), '\n')
    
    fig = ou.plot_levels(test[trading_pair], stop_loss=False)
    
    fig.set_figheight(5)
    fig.set_figwidth(20)  
'''