import numpy as np
from .bbo_agent import BBOAgent
from typing import Callable


class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    initPopulationFunction (function): creates the first generation of
                    individuals to be used for the GA
        input: populationSize (int)
        output: a numpy matrix of size (N x M) where N is the number of 
                individuals in the population and M is the number of 
                parameters (size of the parameter vector)
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, 
                 initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10):
        self._name = "Genetic_Algorithm"
      
        #TODO
        self._initPopulationFunction=initPopulationFunction
        self._populationSize=populationSize
        self._population = self._initPopulationFunction(populationSize) #TODO: set this value to the most recently created 
        self._init_population=self._initPopulationFunction(populationSize)
        self._numElite=numElite
        self._numEpisodes=numEpisodes
        self._evaluationFunction=evaluationFunction
        self._kp=populationSize//2
        self._alpha=0.08
        self.best_J=-float('inf')
        self.best_theta=self._population[0]
        self.G_list=[]
        pass

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        
        return self.best_theta
 
        pass

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        #TODO
        epsilon=np.random.normal(0,1,size=len(parent))
        child=self._alpha*epsilon+parent
        return child
        pass

    def train(self)->np.ndarray:
        #TODO
        
        J=np.zeros(self._populationSize)
        gen=self._population
        for k in range(self._populationSize):
            return_params=self._evaluationFunction(gen[k],self._numEpisodes)
            if type(return_params)!=tuple:
                J[k]=return_params
            else:
                J[k],G=return_params
                self.G_list=np.append(self.G_list,[G])
            #theta_J[k]=np.array([theta_k,J[k]])
        J_sorted=np.argsort(J)[::-1]
        J=J[J_sorted]
        gen=gen[J_sorted]
        parents = gen[np.arange(self._kp)]
        next_gen = gen[np.arange(self._numElite)]
        K_c=self._populationSize-self._numElite
        if J[0]>self.best_J:
            self.best_theta=self._population[0]
            self.best_J=J[0]
        for k in range(K_c):
            sample=np.random.choice(np.arange(self._kp))
            child=self._mutate(parents[sample])
            next_gen=np.vstack([next_gen, child])
        self._population=next_gen
        return self._population[0]
        
        pass

    def reset(self)->None:
        #TODO
     
        self._population=self._init_population
       
        pass