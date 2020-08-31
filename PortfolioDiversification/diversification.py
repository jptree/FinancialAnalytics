__author__ = "Jason Petri"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import itertools
import random
import sys


stock_data = pd.read_csv('../NonNormalReturnOptimization/Data.csv', index_col=0)

n = 13
tickers_list = random.sample(list(stock_data.columns), n)  # careful tweaking this number.
n_port_stdev = {}

total_comb = 2 ** n - 1


def calculate_portfolio_variance(tickers, data):
    num_securities = len(tickers)
    security_weights = 1 / num_securities
    m_weight = np.full((num_securities, 1), security_weights)
    m_cov = data[tickers].cov()
    portfolio_variance = np.dot(np.dot(np.transpose(m_weight), m_cov), m_weight)
    return portfolio_variance


t = 0
for L in range(1, len(tickers_list) + 1):
    port_variances_at_n = []
    for subset in itertools.combinations(tickers_list, L):
        l_subset = list(subset)
        port_val = calculate_portfolio_variance(l_subset, stock_data)
        port_variances_at_n.append(port_val)
        sys.stdout.write('\r')
        sys.stdout.write('Percent Complete:  ' + str(round((t / total_comb) * 100, 2)) + '%')
        sys.stdout.flush()

        t += 1

    n_port_stdev[L] = math.sqrt(float(sum(port_variances_at_n) / len(port_variances_at_n))) * math.sqrt(12)
sys.stdout.write('\n')
sys.stdout.flush()

print(tickers_list)
print(n_port_stdev)


plt.plot(n_port_stdev.keys(), n_port_stdev.values())
plt.xlabel('number of securities')
plt.ylabel('average annual portfolio risk')
plt.title('Average Portfolio Risk Increasing n Stocks in Portfolio')
plt.show()
