"""
Module to implement brownian motion

this module makes it possible to generate an image object motion in a 2D
environment considering a stochastic behaviour.

The motion logic is based on an angle and a motion direction. Then just
the angel must be brownian. Direction is about to be a constant quantity.
"""

import numpy as np

class Brownian():
    """
        A Brownian motion class constructor

        Based on Tirthajyoti Sarkar at
        https://towardsdatascience.com/brownian-motion-with-python-9083ebc46ff0
    """
    def __init__(self, n_step, time_vector, s0=100, mu=0.2, sigma=0.68):
        """
            Init class

            Arguments:
                n_step: total steps to perform
                time_vector: vector of time iterations
                s0: Iniital stock price, default 100
                mu: 'Drift' of the stock (upwards or downwards), default 1
                sigma: 'Volatility' of the stock, default 1
        """

        #self.x0 = 0
        if n_step < 30:
            print("WARNING! The number of steps is small. It may \
                  not generate a good stochastic process sequence!")
        self.n_step = n_step
        self.i = 1 #current time vector position start from 2nd position
        self.time_vector = time_vector
        #self.w = np.zeros(n_step)
        self.prevw = 0
        #self.s = np.zeros(n_step)
        self.s0 = s0
        self.mu = mu
        self.sigma = sigma

    def apply_brownian_eq(self):
        """
            Generate motion by drawing from the Normal distribution 

            Returns:
                w brownian value
        """

        #w = np.ones(self.n_step)*self.x0
        #for i in range(1,self.n_step):
        # Sampling from the Normal distribution
        yi = np.random.normal()
        # Weiner process
        #self.w[i] = self.w[i-1] + (yi/np.sqrt(self.n_step))
        w = self.prevw + (yi/np.sqrt(self.n_step))
        self.prevw = w
        return w

    def geometric_brownian_motion(self, i):
        """
            Models geometric brownian motion  S(t) using the Weiner process W(t) as
            `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`

            Returns:
                s: geometric brownian motion according to the time_vector
        """

        #n_step = int(deltaT/dt)
        #deltaT = self.n_step * dt
        #time_vector = np.linspace(0,deltaT,num=self.n_step)
        # Stock variation
        stock_var = (self.mu - (self.sigma**2/2)) * self.time_vector[i]
        # Forcefully set the initial value to zero for the stock price simulation
        #self.x0=0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = self.sigma * self.apply_brownian_eq()
        # Add two time series, take exponent, and multiply by the initial stock price
        s = self.s0 * (np.exp(stock_var + weiner_process))

        return s



