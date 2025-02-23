import numpy as np
from scipy.interpolate import interp1d
from scipy.stats import norm


def OptionPrice_FDM_Explicit(S0, K, r, sigma, T, Smax, m, n, option_type):
    #initialisation and parameters
    P  = np.zeros((m + 1, n + 1))
    S =  np.linspace(0, Smax, m + 1)
    dS = Smax / m
    dt= T / n
    ii = S / dS

    #boundary conditions
    if option_type == 'call':
        P[m, :] = [Smax - K * np.exp(r * ( n - j) * dt) for j in range(n + 1)]
    if option_type == 'put':
        P[0, :] = [K * np.exp(-r * ( n - j) * dt) for j in range(n + 1)]
    #terminal conditions
    if option_type == 'call':
        P[:, n] = np.array(np.maximum(0, S - K))
    if option_type == 'put':
        P[:, n] = np.array(np.maximum(0, K - S))
        #the coefficients

    a = .5 * (-r * ii + (sigma * ii)**2) * dt
    b = ((sigma * ii)**2 + r) * dt
    c = .5 * (r * ii + (sigma * ii)**2) * dt

    for j in range(n-1, -1, -1):
        for i in range(1, m):
            P[i, j] = a[i] * P[i - 1, j + 1] \
                      + (1 - b[i]) * P[i, j + 1] + c[i] * P[i + 1, j + 1]

        #linear interpolation to get the price
    price = interp1d(S, P[:, 0])
    #return the price with S = S0
    return price(S0)

def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d1)

def binomial_tree(S, K, T, r, sigma, N, option_type):
    """
    S: Stock price
    K: Strike price
    T: Time to expiration (years)
    r: Risk-free rate (decimal)
    sigma: Volatility (decimal)
    N: Number of time steps
    option_type: "call" or "put"
    """
    dt = T / N  # Time step size
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u  # Down factor
    p = (np.exp(r * dt) - d) / (u - d)  # Risk-neutral probability

    # Stock price tree
    stock_tree = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S * (u ** (i - j)) * (d ** j)

    # Option value tree
    option_tree = np.zeros((N + 1, N + 1))

    # Terminal Payoff
    if option_type == "call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    elif option_type == "put":
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)

    # Backward induction to calculate option price
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if option_type == "call":
                exercise_value = stock_tree[j, i] - K
            else:
                exercise_value = K - stock_tree[j, i]
            option_tree[j, i] = max(continuation_value, exercise_value)

    return round(option_tree[0, 0], 4)


