# Monte Carlo Option Pricer

A sophisticated Monte Carlo simulation-based option pricing library that implements advanced variance reduction techniques for accurate European and American option valuations.

## Features

- **Multiple Option Types**: Support for both Call and Put options
- **European & American Style**: Handles both European and American exercise styles
- **Variance Reduction**: Implements antithetic variates and control variate methods
- **Greeks Calculation**: Computes Delta and Gamma using finite differences
- **Dividend Support**: Accounts for dividend yield in pricing calculations
- **Statistical Analysis**: Provides confidence intervals and standard errors

## Installation

Ensure you have the required dependencies:

```bash
pip install numpy scipy
```

## Quick Start

```python
from monte_carlo_pricer import MonteCarloOptionPricer

# Initialize with market parameters
pricer = MonteCarloOptionPricer(
    S0=100.0,      # Current stock price
    K=100.0,       # Strike price
    T=1.0,         # Time to expiration (years)
    r=0.05,        # Risk-free rate
    sigma=0.2,     # Volatility
    q=0.02         # Dividend yield
)

# Price a European call option
result = pricer.price_option(option_type="call", option_style="european")
print(f"Option Price: ${result['price']:.2f}")

# Calculate Greeks
greeks = pricer.calculate_greeks(option_type="call")
print(f"Delta: {greeks['delta']:.4f}")
```

## Class Parameters

### MonteCarloOptionPricer

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `S0` | float | Initial stock price | Required |
| `K` | float | Strike price | Required |
| `T` | float | Time to maturity (years) | Required |
| `r` | float | Risk-free interest rate | Required |
| `sigma` | float | Volatility (annualized) | Required |
| `q` | float | Dividend yield | 0.0 |
| `num_simulations` | int | Number of Monte Carlo paths | 100,000 |
| `num_steps` | int | Time steps per simulation | 252 |

## Methods

### `price_option(option_type, option_style)`

Calculates option price using Monte Carlo simulation with variance reduction.

**Parameters:**
- `option_type`: "call" or "put"
- `option_style`: "european" or "american"

**Returns:**
```python
{
    "price": float,           # Option price
    "std_error": float,       # Standard error
    "conf_interval": tuple    # 95% confidence interval
}
```

### `calculate_greeks(option_type)`

Computes option Greeks using finite difference approximation.

**Parameters:**
- `option_type`: "call" or "put"

**Returns:**
```python
{
    "delta": float,    # Price sensitivity to underlying
    "gamma": float     # Delta sensitivity to underlying
}
```

## Variance Reduction Techniques

### 1. Antithetic Variates
The implementation uses antithetic variates by generating paired random variables (Z, -Z) to reduce variance in the Monte Carlo estimation.

### 2. Control Variates
Uses Black-Scholes prices as control variates to improve convergence:
- Calculates correlation between MC payoffs and final stock prices
- Applies optimal control variate coefficient
- Significantly reduces standard error

## Mathematical Foundation

### Geometric Brownian Motion
Stock price evolution follows:
```
dS = S(r - q)dt + S*σ*dW
```

### Discretization
```
S(t+1) = S(t) * exp((r - q - 0.5*σ²)*dt + σ*√dt*Z)
```

Where Z ~ N(0,1)

### American Option Approximation
For American options, the implementation uses a simple early exercise check at each time step, exercising when intrinsic value is positive.

## Example Output

```
European Call Option:
Price: $7.89
95% Confidence Interval: ($7.85, $7.93)
Standard Error: 0.0204
Delta: 0.5987
Gamma: 0.0193

American Put Option:
Price: $6.74
95% Confidence Interval: ($6.70, $6.78)
Standard Error: 0.0198
Delta: -0.4013
Gamma: 0.0193
```

## Performance Considerations

- **Simulation Count**: Default 100,000 simulations balance accuracy and speed
- **Time Steps**: 252 steps (daily) provide good accuracy for most applications
- **Seed Setting**: Fixed seed (42) ensures reproducible results
- **Memory Usage**: Paths stored in memory; consider reducing `num_simulations` for large-scale applications

## Limitations

1. **American Exercise**: Uses simplified early exercise approximation
2. **Path Dependency**: Not optimized for path-dependent options
3. **Multiple Assets**: Single-asset implementation only
4. **Interest Rates**: Assumes constant risk-free rate

## Theoretical Background

This implementation is based on:
- Black-Scholes-Merton option pricing theory
- Monte Carlo methods in finance
- Variance reduction techniques for improved convergence
- Finite difference methods for Greeks calculation

## Contributing

When contributing:
1. Maintain numerical stability
2. Add comprehensive tests for new features
3. Document mathematical foundations
4. Consider computational efficiency

## References

- Hull, J. C. "Options, Futures, and Other Derivatives"
- Glasserman, P. "Monte Carlo Methods in Financial Engineering"
- Longstaff, F. A., & Schwartz, E. S. "Valuing American Options by Simulation"