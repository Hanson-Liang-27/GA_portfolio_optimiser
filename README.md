# A Genetic Algorithm for Portfolio Optimisation
Markowitz (1952) proposed the foundation of modern portfolio theory with mean-variance portfolio optimisation. Providing a method for investors to calculate how best to allocate and weight their capital to individual shares based upon their comovements.

Genetic Algorithms are optimisation algorithms which attempt to replicate the processes described by Charles Darwin in the theory of natural evolution.

Thus, a GA can be applied to the portfolio optimisation problem. This repository contains a GA object which takes some user inputted stock tickers and returns the optimal weightings. The weights are evaluated by the objective function which is the Sharpe Ratio. The optimal portfolio will be the one which offers the best Sharpe Ratio, meaning the best expected return per unit of risk.

The GA is real-value encoded and utilises uniform crossover and boundary mutation with selection based on the roulette wheel method as well as employing elitism.

The result outputs are not investment advice. There are a series of limitations to this method of portfolio allocation. The program serves as a tool to better understand multi-objective optimisation problems using financial data and is not for investment decisions.
