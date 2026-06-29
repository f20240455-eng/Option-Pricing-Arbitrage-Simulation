# Strike Squad - FIN F311
# Binomial Tree Option Pricing Implementation


"""
Binomial Tree Option Pricing Model

This module implements the Cox-Ross-Rubinstein (CRR) binomial tree model
for option pricing.

The binomial model discretizes time into steps and models the stock price
evolution as a recombining tree. This approach converges to the Black-Scholes
result as the number of steps increases.
"""

import numpy as np
from typing import Literal, Tuple

class BinomialTreeModel:
    """
    Cox-Ross-Rubinstein Binomial Tree for option pricing.
    
    This implementation uses the recombining tree structure for computational
    efficiency. Supports both European and American options.
    """
    
    def __init__(self,
                 stock_price: float,
                 strike_price: float,
                 time_to_expiry: float,
                 risk_free_rate: float,
                 volatility: float,
                 n_steps: int,
                 option_type: Literal['call', 'put'],
                 exercise_type: Literal['european', 'american'] = 'european',
                 dividend_yield: float = 0.0):
        """
        Initialize the binomial tree model.
        
        Parameters:
            stock_price: Current stock price (S₀)
            strike_price: Option strike price (K)
            time_to_expiry: Time to expiration in years (T)
            risk_free_rate: Annual risk-free rate (r)
            volatility: Annual volatility (σ)
            n_steps: Number of time steps in the tree
            option_type: 'call' or 'put'
            exercise_type: 'european' or 'american'
            dividend_yield: Continuous dividend yield (q)
        """
        self.S0 = stock_price
        self.K = strike_price
        self.T = time_to_expiry
        self.r = risk_free_rate
        self.sigma = volatility
        self.n = n_steps
        self.option_type = option_type
        self.exercise_type = exercise_type
        self.q = dividend_yield
        
        # Calculate tree parameters using CRR formulation
        self._calculate_parameters()
    
    def _calculate_parameters(self):
        """
        Calculate binomial tree parameters using CRR methodology.
        
        From Hull (2018), Equations 13.10-13.12:
        Δt = T/n
        u = e^(σ√Δt)
        d = e^(-σ√Δt) = 1/u
        p = (e^((r-q)Δt) - d) / (u - d)
        """
        # Time step
        self.dt = self.T / self.n
        
        # Up and down factors (CRR methodology)
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1 / self.u  # Equivalent to exp(-sigma * sqrt(dt))
        
        # Risk-neutral probability
        # This ensures no-arbitrage condition
        numerator = np.exp((self.r - self.q) * self.dt) - self.d
        denominator = self.u - self.d
        self.p = numerator / denominator
        
        # Discount factor for one time step
        self.discount = np.exp(-self.r * self.dt)
        
        # Validation: probability must be between 0 and 1
        if not (0 < self.p < 1):
            raise ValueError(
                f"Risk-neutral probability {self.p:.4f} is outside valid range (0,1). "
                "This indicates arbitrage opportunities or invalid parameters."
            )
    
    def _build_stock_tree(self) -> np.ndarray:
        """
        Build the stock price tree.
        
        Creates a 2D array where element [i,j] represents the stock price
        at time step i and state j. Uses the recombining tree property
        to reduce memory usage.
        
        Returns:
            2D numpy array of stock prices
        """
        # Initialize array to store stock prices
        # Size: (n+1) x (n+1) to accommodate all nodes
        stock_tree = np.zeros((self.n + 1, self.n + 1))
        
        # Fill the tree
        # At time step i, we have i+1 possible states
        for i in range(self.n + 1):
            for j in range(i + 1):
                # Stock price at node (i,j): S₀ * u^j * d^(i-j)
                stock_tree[i, j] = self.S0 * (self.u ** j) * (self.d ** (i - j))
        
        return stock_tree
    
    def _calculate_terminal_payoffs(self, stock_tree: np.ndarray) -> np.ndarray:
        """
        Calculate option payoffs at expiration (terminal nodes).
        
        Parameters:
            stock_tree: Array of stock prices at all nodes
        
        Returns:
            Array of option values at terminal nodes
        """
        # Terminal stock prices (at time step n)
        terminal_prices = stock_tree[self.n, :]
        
        if self.option_type == 'call':
            # Call payoff: max(S_T - K, 0)
            payoffs = np.maximum(terminal_prices - self.K, 0)
        else:
            # Put payoff: max(K - S_T, 0)
            payoffs = np.maximum(self.K - terminal_prices, 0)
        
        return payoffs
    
    def _backward_induction(self, 
                           stock_tree: np.ndarray,
                           option_tree: np.ndarray) -> np.ndarray:
        """
        Perform backward induction to calculate option values.
        
        Starting from terminal nodes, work backwards to calculate option
        values at each node using risk-neutral valuation.
        
        Parameters:
            stock_tree: Array of stock prices
            option_tree: Array to store option values (initialized with terminal payoffs)
        
        Returns:
            Updated option tree with all values calculated
        """
        # Work backwards from time step n-1 to 0
        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                # Calculate continuation value (discounted expected value)
                # Using risk-neutral probabilities
                continuation = self.discount * (
                    self.p * option_tree[i + 1, j + 1] + 
                    (1 - self.p) * option_tree[i + 1, j]
                )
                
                # For European options, value is just continuation value
                if self.exercise_type == 'european':
                    option_tree[i, j] = continuation
                
                # For American options, check early exercise
                else:
                    if self.option_type == 'call':
                        intrinsic = max(stock_tree[i, j] - self.K, 0)
                    else:
                        intrinsic = max(self.K - stock_tree[i, j], 0)
                    
                    # American option value is max of continuation and early exercise
                    option_tree[i, j] = max(continuation, intrinsic)
        
        return option_tree
    
    def price(self) -> float:
        """
        Calculate the option price using binomial tree.
        
        This is the main method that orchestrates the pricing process:
        1. Build stock price tree
        2. Calculate terminal payoffs
        3. Perform backward induction
        4. Return option value at root node
        
        Returns:
            Option price
        """
        # Step 1: Build stock price tree
        stock_tree = self._build_stock_tree()
        
        # Step 2: Initialize option value tree
        option_tree = np.zeros_like(stock_tree)
        
        # Step 3: Calculate terminal payoffs
        option_tree[self.n, :] = self._calculate_terminal_payoffs(stock_tree)
        
        # Step 4: Backward induction
        option_tree = self._backward_induction(stock_tree, option_tree)
        
        # Step 5: Return option value at root (today)
        return option_tree[0, 0]
    
    def calculate_delta(self) -> float:
        """
        Calculate option delta using the binomial tree.
        
        Delta is approximated using the first two nodes in the tree:
        Δ ≈ (V_up - V_down) / (S_up - S_down)
        
        Returns:
            Delta value
        """
        # Build trees
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        option_tree[self.n, :] = self._calculate_terminal_payoffs(stock_tree)
        option_tree = self._backward_induction(stock_tree, option_tree)
        
        # Calculate delta using nodes at time step 1
        delta = ((option_tree[1, 1] - option_tree[1, 0]) / 
                (stock_tree[1, 1] - stock_tree[1, 0]))
        
        return delta
    
    def calculate_gamma(self) -> float:
        """
        Calculate option gamma using the binomial tree.
        
        Gamma is approximated using the curvature of the option value function.
        
        Returns:
            Gamma value
        """
        # Build trees
        stock_tree = self._build_stock_tree()
        option_tree = np.zeros_like(stock_tree)
        option_tree[self.n, :] = self._calculate_terminal_payoffs(stock_tree)
        option_tree = self._backward_induction(stock_tree, option_tree)
        
        # Calculate gamma using nodes at time steps 1 and 2
        delta_up = ((option_tree[2, 2] - option_tree[2, 1]) / 
                   (stock_tree[2, 2] - stock_tree[2, 1]))
        delta_down = ((option_tree[2, 1] - option_tree[2, 0]) / 
                     (stock_tree[2, 1] - stock_tree[2, 0]))
        
        h = 0.5 * (stock_tree[2, 2] - stock_tree[2, 0])
        gamma = (delta_up - delta_down) / h
        
        return gamma
    
    def get_tree_parameters(self) -> dict:
        """
        Return the calculated tree parameters for verification.
        
        Returns:
            Dictionary with tree parameters
        """
        return {
            'time_step': self.dt,
            'up_factor': self.u,
            'down_factor': self.d,
            'risk_neutral_prob': self.p,
            'discount_factor': self.discount,
            'number_of_steps': self.n
        }


def price_binomial(stock_price: float,
                  strike_price: float,
                  time_to_expiry: float,
                  risk_free_rate: float,
                  volatility: float,
                  n_steps: int,
                  option_type: Literal['call', 'put'],
                  exercise_type: Literal['european', 'american'] = 'european',
                  dividend_yield: float = 0.0) -> float:
    """
    Convenience function for binomial option pricing.
    
    Parameters:
        stock_price: Current stock price
        strike_price: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free rate
        volatility: Annual volatility
        n_steps: Number of time steps
        option_type: 'call' or 'put'
        exercise_type: 'european' or 'american'
        dividend_yield: Continuous dividend yield
    
    Returns:
        Option price
    """
    model = BinomialTreeModel(
        stock_price, strike_price, time_to_expiry, risk_free_rate,
        volatility, n_steps, option_type, exercise_type, dividend_yield
    )
    
    return model.price()


def analyze_convergence(stock_price: float,
                       strike_price: float,
                       time_to_expiry: float,
                       risk_free_rate: float,
                       volatility: float,
                       steps_list: list,
                       option_type: Literal['call', 'put'],
                       dividend_yield: float = 0.0) -> dict:
    """
    Analyze binomial model convergence for different step counts.
    
    This function prices the same option using different numbers of steps
    to demonstrate convergence to the Black-Scholes result.
    
    Parameters:
        stock_price: Current stock price
        strike_price: Option strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free rate
        volatility: Annual volatility
        steps_list: List of step counts to test
        option_type: 'call' or 'put'
        dividend_yield: Continuous dividend yield
    
    Returns:
        Dictionary with convergence results
    """
    results = {
        'steps': [],
        'binomial_price': [],
        'computation_time': []
    }
    
    import time
    
    for n_steps in steps_list:
        start_time = time.time()
        
        price = price_binomial(
            stock_price, strike_price, time_to_expiry, risk_free_rate,
            volatility, n_steps, option_type, 'european', dividend_yield
        )
        
        computation_time = time.time() - start_time
        
        results['steps'].append(n_steps)
        results['binomial_price'].append(price)
        results['computation_time'].append(computation_time)
    
    return results