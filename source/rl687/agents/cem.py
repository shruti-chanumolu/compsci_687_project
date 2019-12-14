import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):
        #TODO
        self._name = "Cross_Entropy_Method"
        self._theta = theta #TODO: set this value to the current mean parameter vector
        self._Sigma = sigma*np.identity(len(self._theta)) #TODO: set this value to the current covariance matrix
        self.in_theta=theta
        self.in_Sigma=sigma*np.identity(len(self._theta))
        self._popSize=popSize
        self._numElite=numElite
        self._numEpisodes=numEpisodes
        self._evaluationFunction=evaluationFunction
        self._epsilon=epsilon
        self._best_J=-float('inf')
        self._best_theta=self.in_theta
        self.G_list=[]
        pass
        

    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        #TODO
        return self._best_theta  
        pass

    def train(self)->np.ndarray:
        #TODO
        theta_J=np.zeros((self._popSize,len(self._theta)))
        J=np.zeros(self._popSize)
        for k in range(self._popSize):
            theta_k=np.array(np.random.multivariate_normal(self._theta,self._Sigma))
            return_params=self._evaluationFunction(theta_k,self._numEpisodes)
            print(type,type(return_params))
            if type(return_params)!=tuple:
                J_k=return_params
            else:
                J_k,G=return_params
                self.G_list=np.append(self.G_list,[G])
            theta_J[k]=theta_k.reshape((1,len(theta_k)))
            J[k]=J_k  
        
        J_sorted=np.argsort(J)[::-1]
        theta_J=theta_J[J_sorted]
        if self._best_J<J[J_sorted[0]]:
            self._best_J=J[J_sorted[0]]
            self._best_theta=theta_J[0]
            
        theta_k_e=theta_J[np.arange(self._numElite)]
        self._theta=np.sum(theta_k_e,axis=0)/self._numElite
        diff=theta_k_e-self._theta.reshape((1,len(self._theta))) 
        self._Sigma=(self._epsilon*(np.identity(len(self._theta)))+(np.dot(diff.T,diff)))/(self._epsilon+self._numElite)
        
        return theta_J[0]

    def reset(self)->None:
        #TODO
        self._theta=self.in_theta
        print(self._theta,np.dtype(self._theta))
        self._Sigma=self.in_Sigma
        pass
