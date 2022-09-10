import numpy as np
import pandas as pd
import random

class genetic_algorithm():

    def __init__(
        self, 
        data: pd.DataFrame,
        risk_free_rate: float, 
        population: int, 
        generations: int, 
        crossover_rate: float, 
        mutation_rate: float, 
        elite_rate: float) -> None:

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
        self.optimal_weight = 0 
        self.estimated_return = 0 
        self.estimated_risk = 0 


    def optimise_weights(self) -> None:
        
        weights = self.generate_weights()

        for i in range(0, self.generations):
            results = self.fitness_func(weights = weights)

            elites, parents = self.elitism(results)
            parents, no_cross_parents = self.selection(parents)
            children = self.crossover(parents)
            children = self.mutation(children) 
            
            weights = self.next_gen(elites, children, no_cross_parents)
            
            avg_res = self.avg_gen_result(weights)
            print('Generation', i, ': Average Sharpe Ratio of', avg_res, 'from', len(weights), 'chromosomes')
            
        self.optimal_solution(weights)


    def generate_weights(self) -> np.array:

        weight_array = np.empty((self.population, (self.n_assets + 2)))
        weights = []
        
        for i in range(0, self.population):
            weighting = np.random.random(self.n_assets)
            weighting /= np.sum(weighting)
            weights.append(weighting)
        weights = np.array(weights)
        
        for i in range(0, self.n_assets):
            weight_array[:, i] = weights[:, i]
        
        return weight_array


    def fitness_func(self, weights: np.array) -> np.array:

        fitness = []
        
        for i in range(0, len(weights)):
            w_return = (weights[i, 0:self.n_assets] * self.port_return) 
            w_risk = np.sqrt(np.dot(weights[i, 0:self.n_assets].T, np.dot(self.port_risk, weights[i, 0:self.n_assets]))) * np.sqrt(252)
            score = ((np.sum(w_return) * 100) - self.rf) / (np.sum(w_risk) * 100)
            fitness.append(score)
            
        fitness = np.array(fitness).reshape(len(weights))
        weights[:, self.n_assets] = fitness
        
        return weights


    def elitism(self, fit_func_res):
        sorted_ff = fit_func_res[fit_func_res[:, self.n_assets].argsort()]
        elite_w = int(len(sorted_ff) * self.elite_rate)
        elite_results = sorted_ff[-elite_w:]
        non_elite_results = sorted_ff[:-elite_w] 
        
        return elite_results, non_elite_results


    def selection(self, parents):     
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


    def crossover(self, weights):   
        for i in range(0, int((len(weights))/2), 2): 
            gen1, gen2 = weights[i], weights[i+1]
            gen1, gen2 = self.uni_co(gen1, gen2)
            weights[i], weights[i+1] = gen1, gen2
            
        weights = self.normalise(weights)
        
        return weights
        

    def uni_co(self, gen1, gen2):
        prob = np.random.normal(1, 1, self.n_assets)
        
        for i in range(0, len(prob)):
            if prob[i] > self.crossover_rate:
                gen1[i], gen2[i] = gen2[i], gen1[i]  
                
        return gen1, gen2


    def mutation(self, generation): 

        weight_n = len(generation) * ((np.shape(generation)[1]) - 2)
        mutate_gens = int(weight_n * self.mutation_rate)
        
        if (mutate_gens < 1):
            return generation
        
        else:
            for i in range(0, mutate_gens):

                rand_pos_x, rand_pos_y = random.randint(0, (len(generation) - 1)), random.randint(0, (self.n_assets - 1))
                mu_gen = generation[rand_pos_x][rand_pos_y]
                mutated_ind = mu_gen * np.random.normal(0,1)
                generation[rand_pos_x][rand_pos_y] = abs(mutated_ind)
                generation = self.normalise(generation)

            return generation
            


    def find_series_mean(self) -> np.array:
        returns = np.log(self.data / self.data.shift(1))
        port_return = np.array(returns.mean() * 252)
        return port_return 


    def find_series_sd(self) -> np.array:
        returns = np.log(self.data / self.data.shift(1))
        port_risk = returns.cov()
        return port_risk


    def normalise(self, generation):
        for i in range(0, len(generation)):
            generation[i][0:self.n_assets] /= np.sum(generation[i][0:self.n_assets])
        return generation


    def next_gen(self, elites, children, no_cross_parents) -> np.array:
        weights = np.vstack((elites, children, no_cross_parents))
        return weights 


    def optimal_solution(self, generations: int) -> np.array:
        optimal_weights = generations[generations[:, (self.n_assets + 1)].argsort()]
        self.optimal_weight = optimal_weights[:, 0:self.n_assets][0]
        self.estimated_return = optimal_weights[:, (self.n_assets)][0]
        self.estimated_risk = optimal_weights[:, (self.n_assets + 1)][0]


    def avg_gen_result(self, weights) -> float:
        average = round(np.mean(weights[:, self.n_assets]), 2)
        return average
