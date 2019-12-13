import numpy as np
from .skeleton import Policy
from typing import Union
from itertools import product 


class LinearSoftmax(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self,numActions: int,order, State_dims,alpha,theta,sigma=1):
        
        #The internal policy parameters must be stored as a matrix of size
        #(numStates x numActions)
        #TODO
        self._order=order
        self._numActions=numActions
        self._sigma=sigma
        self._theta=theta
        self._alpha=alpha
        self.C=np.matrix(list(product(range(order+1), repeat=State_dims)))
        #self.C=np.asarray(list(product(range(order+1), repeat=State_dims)))
        self.feature_size=(order+1)**State_dims
        self.fourier_base=np.zeros(self.feature_size)
        pass
    
    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)
    
    
    def fourier_features(self,state):
        #normalising states
        value_func=np.cos(np.pi*np.dot(self.C,state))
        #print(value_func)
        self.fourier_base=value_func.reshape((self.feature_size,1))
        #print(self.fourier_base)
    
    
    def score(self,action):
        #print(self._theta[action])
        theta_2d=self._theta.reshape(self._numActions,-1)
        return np.dot(theta_2d[action],self.fourier_base)
    
    def policy(self,state,action):
        self.fourier_features(state)
        score=self.score(action)
        exp_score=np.exp(score)
        exp_score_ax=np.exp(self.score(abs(action-1)))
        softmax_probs=exp_score/(exp_score+exp_score_ax)
        #print(softmax_probs.shape)
        return softmax_probs



    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        
        #TODO
        if action!=None:
            return self.linear_softmax_probs(state)[action]
        return self.linear_softmax_probs(state)
        pass

    def samplAction(self, state:int)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        
        #TODO
        return np.random.choice(np.arange(self._numActions),1,p=self.linear_softmax_probs(state))[0]
        
        pass

