import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize_scalar
from scipy.stats import norm
from scipy.optimize import newton

class opciones:
    def __init__(self):
        return None

    def black_scholes_with_dividends(self,S, K, T, r, q, sigma, option_type='C'):
        """
        Calculates the Black-Scholes price of a European option with continuous dividends.

        Args:
            S: Current price of the underlying asset.
            K: Strike price of the option.
            T: Time to expiration (in years).
            r: Risk-free interest rate (annualized, continuously compounded).
            q: Continuous dividend yield (annualized).
            sigma: Volatility of the underlying asset.
            option_type: 'call' or 'put'.

        Returns:
            The Black-Scholes price of the option.
        """

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'C':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == 'P':
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        return price
    

    def implied_volatility_with_dividends(self, S, K, T, r, q, market_price, option_type='C', initial_guess=0.2, tolerance=1e-6, max_iterations=1000):
        """
        Calculates the implied volatility of a European option with continuous dividends
        using the Black-Scholes model and the Newton-Raphson method.

        Args:
            S: Current price of the underlying asset.
            K: Strike price of the option.
            T: Time to expiration (in years).
            r: Risk-free interest rate (annualized, continuously compounded).
            q: Continuous dividend yield (annualized).
            market_price: The observed market price of the option.
            option_type: 'call' or 'put'.
            initial_guess: Initial guess for the implied volatility.  A good initial guess is crucial for convergence.
            tolerance: The desired accuracy (difference between market price and calculated price).
            max_iterations: The maximum number of iterations allowed.

        Returns:
            The implied volatility (or None if the algorithm fails to converge).
        """

        try:
            # Use the Newton-Raphson method to find the implied volatility.
            implied_vol = newton(
                lambda sigma: self.black_scholes_with_dividends(S, K, T, r, q, sigma, option_type) - market_price,
                x0=initial_guess,  # Initial guess for volatility
                fprime=lambda sigma: self.vega_with_dividends(S, K, T, r, q, sigma), # Use the vega function as the derivative
                tol=tolerance,
                maxiter=max_iterations,
                full_output=False,  # We only want the root (implied volatility)
                disp=False # Don't print convergence messages
            )
            # print("vol")
            # print(implied_vol)
            if implied_vol > 0: #check for negative or zero volatility
                return implied_vol
            else:
                return None  # or raise an exception

        except (RuntimeError, OverflowError, ValueError):
            # Handle cases where the Newton-Raphson method fails to converge.
            #  RuntimeError:  Newton-Raphson didn't converge.
            #  OverflowError:  Numbers got too large (often a sign of a bad initial guess or market price).
            #  ValueError: Can occur if sigma goes negative during iteration, resulting in NaN in the calculations.
            return None
        
    def vega_with_dividends(self, S, K, T, r, q, sigma):
        """
        Calculates the vega (sensitivity to volatility) of a European option
        with continuous dividends.

        Args:
            S: Current price of the underlying asset.
            K: Strike price of the option.
            T: Time to expiration (in years).
            r: Risk-free interest rate (annualized).
            q: Continuous dividend yield (annualized).
            sigma: Volatility of the underlying asset.

        Returns:
            The vega of the option.
        """
        
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega
    
    def delta_american_dividend(self, S, K, T, r, q, sigma, n=100):
        """
        Calculates the delta of an American call option with dividends using a binomial tree.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (in years).
            r: Risk-free interest rate (annual).
            q: Dividend yield (annual).
            sigma: Volatility (annual).
            n: Number of time steps in the binomial tree.

        Returns:
            The delta of the American call option.
        """

        dt = T / n
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)

        # Calculate option values (same as in black_scholes_american_dividend)
        option_values = np.maximum(np.zeros(n + 1), S * (u ** np.arange(n, -1, -1)) * (d ** np.arange(0, n + 1)) - K)

        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                stock_price = S * (u ** (i - j)) * (d ** j)
                intrinsic_value = max(0, stock_price - K)
                continuation_value = np.exp(-r * dt) * (p * option_values[j] + (1 - p) * option_values[j + 1])
                option_values[j] = max(intrinsic_value, continuation_value)

        # Delta Calculation:
        delta = (option_values[0] - option_values[1]) / (S * (u - d))  # Central difference approach

        return delta
    
    def european_call_delta_dividend(self, S, K, T, r, q, sigma):
        """
        Calculates the delta of a European call option with dividends.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (in years).
            r: Risk-free interest rate (annual).
            q: Dividend yield (annual).
            sigma: Volatility (annual).

        Returns:
            The delta of the European call option.
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = np.exp(-q * T) * norm.cdf(d1)
        return delta


    def european_put_delta_dividend(self, S, K, T, r, q, sigma):
        """
        Calculates the delta of a European put option with dividends.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (in years).
            r: Risk-free interest rate (annual).
            q: Dividend yield (annual).
            sigma: Volatility (annual).

        Returns:
            The delta of the European put option.
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
        return delta
    
    def gamma_call_dividend(self, S, K, T, r, q, sigma):
        """
        Calculates the gamma of a European call option with dividends.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (in years).
            r: Risk-free interest rate (annual).
            q: Dividend yield (annual).
            sigma: Volatility (annual).

        Returns:
            The gamma of the call option.
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))
        return gamma
    
    def theta_call_dividend(self, S, K, T, r, q, sigma):
        """Calculates theta for a European call option with dividends."""
        #print(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        theta = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - \
                r * K * np.exp(-r * T) * norm.cdf(d2) + \
                q * S * np.exp(-q * T) * norm.cdf(d1)
        return theta

    def theta_put_dividend(self, S, K, T, r, q, sigma):
        """Calculates theta for a European put option with dividends."""
        # print(S)
        # print(K)
        # print(T)
        # print(r)
        # print(q)
        # print(sigma)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        theta = -S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + \
                r * K * np.exp(-r * T) * norm.cdf(-d2) - \
                q * S * np.exp(-q * T) * norm.cdf(-d1)
        return theta

    def vega_european_dividend(self,S, K, T, r, q, sigma):
        """
        Calculates the Vega of a European call or put option with dividends.

        Args:
            S: Current stock price.
            K: Strike price.
            T: Time to maturity (in years).
            r: Risk-free interest rate (annual).
            q: Dividend yield (annual).
            sigma: Volatility (annual).

        Returns:
            The Vega of the option.
        """
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        return vega


