import numpy as np
from scipy.stats import norm

class MonteCarloOptionPricer:
    def __init__(self, S0, K, T, r, sigma, q=0.0, num_simulations=100000, num_steps=252):
        """
        Initialize Monte Carlo option pricer.
        
        Parameters:
        S0 : float : Initial stock price
        K : float : Strike price
        T : float : Time to maturity (in years)
        r : float : Risk-free rate
        sigma : float : Volatility
        q : float : Dividend yield
        num_simulations : int : Number of Monte Carlo paths
        num_steps : int : Number of time steps
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.q = q
        self.num_simulations = num_simulations
        self.num_steps = num_steps
        self.dt = T / num_steps
        np.random.seed(42)

    def _generate_paths(self, antithetic=True):
        """Generate stock price paths using Geometric Brownian Motion."""
        paths = np.zeros((self.num_simulations, self.num_steps + 1))
        paths[:, 0] = self.S0
        
        if antithetic:
            # Generate antithetic variates
            Z = np.random.normal(0, 1, (self.num_simulations // 2, self.num_steps))
            Z = np.concatenate([Z, -Z], axis=0)
        else:
            Z = np.random.normal(0, 1, (self.num_simulations, self.num_steps))
            
        drift = (self.r - self.q - 0.5 * self.sigma**2) * self.dt
        diffusion = self.sigma * np.sqrt(self.dt)
        
        for t in range(1, self.num_steps + 1):
            paths[:, t] = paths[:, t-1] * np.exp(drift + diffusion * Z[:, t-1])
            
        return paths

    def _black_scholes_price(self, option_type):
        """Calculate Black-Scholes price for control variate."""
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        
        if option_type.lower() == "call":
            price = self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1)
        return price

    def price_option(self, option_type="call", option_style="european"):
        """
        Price option using Monte Carlo with variance reduction.
        
        Returns:
        dict : Option price, standard error, and confidence interval
        """
        paths = self._generate_paths(antithetic=True)
        
        if option_type.lower() == "call":
            payoff = lambda S, K: np.maximum(S - K, 0)
        else:
            payoff = lambda S, K: np.maximum(K - S, 0)
            
        if option_style.lower() == "european":
            payoffs = payoff(paths[:, -1], self.K)
        else:  # American option
            payoffs = np.zeros(self.num_simulations)
            for i in range(self.num_simulations):
                for t in range(self.num_steps + 1):
                    if payoff(paths[i, t], self.K) > 0:
                        payoffs[i] = payoff(paths[i, t], self.K) * np.exp(-self.r * t * self.dt)
                        break
                        
        # Control variate adjustment
        bs_price = self._black_scholes_price(option_type)
        mc_mean = np.mean(payoffs)
        cov_xy = np.mean(payoffs * paths[:, -1]) - np.mean(payoffs) * np.mean(paths[:, -1])
        var_x = np.var(paths[:, -1])
        beta = cov_xy / var_x if var_x != 0 else 0
        adjusted_payoffs = payoffs - beta * (paths[:, -1] - self.S0 * np.exp((self.r - self.q) * self.T))
        
        # Calculate price and statistics
        price = np.exp(-self.r * self.T) * np.mean(adjusted_payoffs)
        std_err = np.std(adjusted_payoffs) / np.sqrt(self.num_simulations)
        conf_interval = (price - 1.96 * std_err, price + 1.96 * std_err)
        
        return {
            "price": price,
            "std_error": std_err,
            "conf_interval": conf_interval
        }

    def calculate_greeks(self, option_type="call"):
        """Calculate Delta and Gamma using finite differences."""
        delta_S = 0.01 * self.S0
        
        # Calculate base price
        base_result = self.price_option(option_type)
        base_price = base_result["price"]
        
        # Calculate price with increased S0 for Delta
        self.S0 += delta_S
        up_result = self.price_option(option_type)
        up_price = up_result["price"]
        self.S0 -= delta_S
        
        # Delta
        delta = (up_price - base_price) / delta_S
        
        # Calculate price with decreased S0 for Gamma
        self.S0 -= delta_S
        down_result = self.price_option(option_type)
        down_price = down_result["price"]
        self.S0 += delta_S
        
        # Gamma
        gamma = (up_price - 2 * base_price + down_price) / (delta_S**2)
        
        return {"delta": delta, "gamma": gamma}

# Example usage
if __name__ == "__main__":
    # Example parameters
    S0 = 100.0  # Initial stock price
    K = 100.0   # Strike price
    T = 1.0     # Time to maturity (1 year)
    r = 0.05    # Risk-free rate (5%)
    sigma = 0.2 # Volatility (20%)
    q = 0.02    # Dividend yield (2%)
    
    pricer = MonteCarloOptionPricer(S0, K, T, r, sigma, q)
    
    # Price European and American options
    for style in ["european", "american"]:
        for opt_type in ["call", "put"]:
            result = pricer.price_option(option_type=opt_type, option_style=style)
            greeks = pricer.calculate_greeks(option_type=opt_type)
            print(f"\n{style.capitalize()} {opt_type.capitalize()} Option:")
            print(f"Price: ${result['price']:.2f}")
            print(f"95% Confidence Interval: (${result['conf_interval'][0]:.2f}, ${result['conf_interval'][1]:.2f})")
            print(f"Standard Error: ${result['std_error']:.4f}")
            print(f"Delta: {greeks['delta']:.4f}")
            print(f"Gamma: {greeks['gamma']:.4f}")