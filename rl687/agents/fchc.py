import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """
    
    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        self._name = "First_Choice_Hill_Climbing"
        #TODO
        self._theta = theta #TODO: set this value to the current mean parameter vector
        self._Sigma = sigma*np.identity(len(self._theta)) #TODO: set this value to the current covariance matrix
        self.in_theta=theta
        self.in_Sigma=sigma*np.identity(len(self._theta))
        self._numEpisodes=numEpisodes
        self._evaluationFunction=evaluationFunction
        return_params=self._evaluationFunction(self._theta,self._numEpisodes)
        if type(return_params)!=tuple:
            self._J=return_params
        else:
            self._J,G=return_params
            self.G_list=G
        self.best_J=self._J
        self.best_theta=theta
        
        pass

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self.best_theta
        pass

    def train(self)->np.ndarray:
        #TODO
        theta_k=np.array(np.random.multivariate_normal(self._theta,self._Sigma))
        return_params=self._evaluationFunction(theta_k,self._numEpisodes)
        if type(return_params)!=tuple:
            J_k=return_params
        else:
            J_k,G=return_params
            self.G_list=np.append(self.G_list,[G])
        if J_k>self._J:
            self._theta=theta_k
            self._J=J_k
        if self._J>self.best_J:
            self.best_theta=self._theta
            self.best_J=self._J 
        return self._theta   
        pass

    def reset(self)->None:
        #TODO
        self._theta=self.in_theta
        print(self._theta,np.dtype(self._theta))
        self._Sigma=self.in_Sigma
        return_params=self._evaluationFunction(self._theta,self._numEpisodes)
        
        if type(return_params)!=tuple:
             self._J=return_params
        else:
            self._J,G=return_params
            self.G_list=np.append(self.G_list,[G])
        pass