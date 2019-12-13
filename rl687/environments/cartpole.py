import numpy as np
from typing import Tuple
from .skeleton import Environment


class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        self._name = "Cartpole"
        
        # TODO: properly define the variables below
        self._action = None
        self._reward = 0
        self._isEnd = False
        self._gamma = 1.0

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep
        self._t = 0.0  # total time elapsed  NOTE: you must use this variable
        

    @property
    def name(self)->str:
        return self._name

    @property
    def reward(self) -> float:
        # TODO
        return self._reward
        pass

    @property
    def gamma(self) -> float:
        # TODO
        return self._gamma
        pass

    @property
    def action(self) -> int:
        # TODO
        return self._action
        pass

    @property
    def isEnd(self) -> bool:
        # TODO
        return self._isEnd
        pass

    @property
    def state(self) -> np.ndarray:
        # TODO
        return np.array([self._x,self._v,self._theta,self._dtheta])
        pass

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        # TODO
        self._action=action
        m=self._mc+self._mp
        F=-10 if action==0 else 10
        x_dot=state[1]
        theta_dot=state[3]
        dtheta_dot=(self._g*np.sin(state[2])+np.cos(state[2])*((-F-(self._mp*self._l*(state[3]**2)*np.sin(state[2])))/m))/(self._l*((4.0/3.0)-(self._mp*(np.cos(state[2])**2)/m)))
                    
        v_dot=(F+self._mp*self._l*((theta_dot**2)*np.sin(state[2])-dtheta_dot*np.cos(state[2])))/m
        next_state=state+self._dt*np.array([x_dot,v_dot,theta_dot,dtheta_dot])
        
        return next_state
        pass

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        # TODO
        return 1

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        # TODO
        self._t+=self._dt
        self._isEnd = self.terminal()
        self._action=action
        next_state = self.nextState(self.state, action)
        self._reward = self.R(self.state, action, next_state)
        self._x=next_state[0]  # horizontal position of cart
        self._v = next_state[1]   # horizontal velocity of the cart
        self._theta = next_state[2]   # angle of the pole
        self._dtheta = next_state[3] 
        self._isEnd = self.terminal()
        return (self.state, self.reward, self.isEnd)
        
        pass

    def reset(self) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        # TODO
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0
        self._action = None
        self._reward = 0
        self._isEnd = False
        self._t=0
            
        pass

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        # TODO
        if (self._t>20) or (abs(self._theta)>(np.pi/12)) or (abs(self._x)>=3):
            return True 
        else: return False
            
        pass
