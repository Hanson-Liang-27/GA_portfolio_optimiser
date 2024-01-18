
# Function Import 
from src.data_loader import * 
from src.genetic_algo_class import * 

# test_for_git_command

# Config
tickers = ['AAPL', 'TSLA', 'MSFT']
population = 500
risk_free_rate = 0.02
generations = 50
crossover_rate = 0.4
mutation_rate = 0.01
elite_rate = 0.25

# Load Data
stock_data = fetch_data(
    tickers = tickers, 
    start_date = '2019-02-01',
    end_date = '2020-12-01'
    )


# Initialise GA
ga = genetic_algorithm(
    data = stock_data,
    risk_free_rate = risk_free_rate, 
    population = population, 
    generations = generations, 
    crossover_rate = crossover_rate, 
    mutation_rate = mutation_rate, 
    elite_rate = elite_rate
)

# Run GA
ga.optimise_weights()

# View Optimal Weights
ga.optimal_weight

# Plot Efficient Frontier
ga.plot_efficient_frontier()
