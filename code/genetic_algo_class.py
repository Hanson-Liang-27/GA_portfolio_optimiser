import numpy as np
import pandas as pd
import random

class genetic_algorithm():
    """
    Genetic Algorithm Implementation. 
    """

    def __init__(
        self, 
        data: pd.DataFrame,
        risk_free_rate: float, 
        population: int, 
        generations: int, 
        crossover_rate: float, 
        mutation_rate: float, 
        elite_rate: float) -> None:
        """
        Initialise the class with data and parameters.
        """

        self.data = data
        self.n_assets = len(self.data.columns)
        self.rf = risk_free_rate
        self.population = population
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_rate = elite_rate
        self.port_risk = self.find_series_sd()
        self.port_return = self.find_series_mean()
        self.weights = 0 
        self.optimal_weight = 0 
        self.estimated_return = 0 
        self.estimated_risk = 0 


    def optimise_weights(self) -> None:
        """
        Run the GA and optimise the weighting.
        """
        self.generate_weights()
        
        for i in range(0, self.generations):
            
            self.fitness_func()
            
            elites, parents = self.elitism()
            
            parents, no_cross_parents = self.selection(parents)

            children = self.crossover(parents)

            children = self.mutation(children) 

            self.next_gen(elites, children, no_cross_parents)
            
            avg_res = self.avg_gen_result()
            print('Generation', i, ': Average Sharpe Ratio of', avg_res, 'from', len(self.weights), 'chromosomes')
            
        self.optimal_solution()


    def generate_weights(self) -> None:
        """
        Generate random weights for each ticker.
        """
        weight_array = np.empty((self.population, (self.n_assets + 2)))
        weights = []
        
        for i in range(0, self.population):
            weighting = np.random.random(self.n_assets)
            weighting /= np.sum(weighting)
            weights.append(weighting)
        weights = np.array(weights)
        
        for i in range(0, self.n_assets):
            weight_array[:, i] = weights[:, i]
        
        self.weights = weight_array


    def fitness_func(self) -> None:
        """
        Evaluate weights by fitness function.
        """ 
        fitness = []
        
        for i in range(0, len(self.weights)):
            w_return = (self.weights[i, 0:self.n_assets] * self.port_return) 
            w_risk = np.sqrt(np.dot(self.weights[i, 0:self.n_assets].T, np.dot(self.port_risk, self.weights[i, 0:self.n_assets]))) * np.sqrt(252)
            score = ((np.sum(w_return) * 100) - self.rf) / (np.sum(w_risk) * 100)
            fitness.append(score)
            
        fitness = np.array(fitness).reshape(len(self.weights))
        self.weights[:, self.n_assets] = fitness


    def elitism(self) -> tuple(np.array, np.array):
        """
        Perform elitism step.

        Returns
        ----------
        elite_results : array of population selected as elites
        non_elite_results : array of population not selected as elites
        """ 
        sorted_ff = self.weights[self.weights[:, self.n_assets].argsort()]
        elite_w = int(len(sorted_ff) * self.elite_rate)
        elite_results = sorted_ff[-elite_w:]
        non_elite_results = sorted_ff[:-elite_w] 
        
        return elite_results, non_elite_results


    def selection(self, parents: np.array) -> tuple(np.array, np.array):
        """
        Perform selection step.

        Parameters 
        ----------
        parents : array of population eligible for selection

        Returns
        ----------
        crossover_gen : array of population after selection now for crossover
        non_crossover_gen : array of population after selection not for crossover
        """ 
        sol_len = int(len(parents) / 2)
        if (sol_len % 2) != 0: sol_len = sol_len + 1
        crossover_gen = np.empty((0, (self.n_assets + 2)))  
        
        for i in range(0, sol_len):
            parents[:, (self.n_assets + 1)] = np.cumsum(parents[:, self.n_assets]).reshape(len(parents))
            rand = random.randint(0, int(sum(parents[:, self.n_assets])))
            
            for i in range(0, len(parents)): nearest_val = min(parents[i:, (self.n_assets + 1)], key = lambda x: abs(x - rand))
            val = np.where(parents == nearest_val)
            index = val[0][0]
            
            next_gen = parents[index].reshape(1, (self.n_assets + 2))
            
            crossover_gen = np.append(crossover_gen, next_gen, axis = 0) 
            parents = np.delete(parents, (val[0]), 0)
            
        non_crossover_gen = crossover_gen.copy()
        
        return crossover_gen, non_crossover_gen


    def crossover(self, weights: np.array) -> np.array:   
        """
        Perform crossover step.

        Parameters 
        ----------
        weights : array of population to be crossed

        Returns
        ----------
        weights : array of population after crossover
        """
        for i in range(0, int((len(weights))/2), 2): 
            gen1, gen2 = weights[i], weights[i+1]
            gen1, gen2 = self.uni_co(gen1, gen2)
            weights[i], weights[i+1] = gen1, gen2
            
        weights = self.normalise(weights)
        
        return weights
        

    def uni_co(self, gen1: np.array, gen2: np.array) -> tuple(np.array, np.array):
        """
        Perform uniform crossover step.

        Parameters 
        ----------
        gen1 : first array of population to be crossovered
        gen2 : second array of population to be crossovered

        Returns
        ----------
        gen1 : first array of population after crossover
        gen2 : second array of population after crossover
        """
        prob = np.random.normal(1, 1, self.n_assets)
        
        for i in range(0, len(prob)):
            if prob[i] > self.crossover_rate:
                gen1[i], gen2[i] = gen2[i], gen1[i]  
                
        return gen1, gen2


    def mutation(self, generation: np.array) -> np.array: 
        """
        Perform mutation step.

        Parameters 
        ----------
        generation : array of population to be mutated

        Returns
        ----------
        generation : mutated population
        """

        weight_n = len(generation) * ((np.shape(generation)[1]) - 2)
        mutate_gens = int(weight_n * self.mutation_rate)
        
        if (mutate_gens < 1):
            return generation
        
        else:
            for _ in range(0, mutate_gens):

                rand_pos_x, rand_pos_y = random.randint(0, (len(generation) - 1)), random.randint(0, (self.n_assets - 1))
                mu_gen = generation[rand_pos_x][rand_pos_y]
                mutated_ind = mu_gen * np.random.normal(0,1)
                generation[rand_pos_x][rand_pos_y] = abs(mutated_ind)
                generation = self.normalise(generation)

            return generation
            


    def find_series_mean(self) -> np.array:
        """
        Compute each tickers time series mean return.

        Returns
        ----------
        port_return : array of ticker mean return 
        """
        returns = np.log(self.data / self.data.shift(1))
        port_return = np.array(returns.mean() * 252)
        return port_return 


    def find_series_sd(self) -> np.array:
        """
        Compute each tickers time series standard deviation.

        Returns
        ----------
        port_risk : array of ticker standard deviations
        """
        returns = np.log(self.data / self.data.shift(1))
        port_risk = returns.cov()
        return port_risk


    def normalise(self, weights: np.array) -> np.array:
        """
        Normalise array.

        Parameters 
        ----------
        weights : array passed to be normalised

        Returns
        ----------
        weights : normalised array
        """
        for i in range(0, len(weights)):
            weights[i][0:self.n_assets] /= np.sum(weights[i][0:self.n_assets])
        return weights 
        


    def next_gen(self, elites: np.array, children: np.array, no_cross_parents: np.array) -> None:
        """
        Stack all selected populations to create next generate population

        Parameters
        ---------- 
        elites : array of population with elites
        children : array of population with crossover children
        no_cross_parents : array of population with no crossover
        """
        self.weights = np.vstack((elites, children, no_cross_parents))


    def optimal_solution(self) -> None:
        """
        Find the optimal solution.
        """
        optimal_weights = self.weights[self.weights[:, (self.n_assets + 1)].argsort()]
        self.optimal_weight = optimal_weights[:, 0:self.n_assets][0]
        self.estimated_return = optimal_weights[:, (self.n_assets)][0]
        self.estimated_risk = optimal_weights[:, (self.n_assets + 1)][0]


    def avg_gen_result(self) -> float:
        """
        Compute average result from current population weights.

        Returns
        ----------
        average : average result 
        """
        return round(np.mean(self.weights[:, self.n_assets]), 2)

