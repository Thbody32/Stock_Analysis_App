import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from scipy.interpolate import interp1d
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns

def garch_model(data):
    model = arch_model(data, vol='Garch', p=1, q=1)
    model_fit = model.fit(disp="off")
    st.write("GARCH Model Summary:")
    st.write(model_fit.summary())

# Function for EGARCH model
def egarch_model(data):
    model = arch_model(data, vol='EGarch', p=1, q=1)
    model_fit = model.fit(disp="off")
    st.write("EGARCH Model Summary:")
    st.write(model_fit.summary())

# Function for QQ Plot
def qq_plot(data):
    fig= plt.figure()
    sm.qqplot(data, line ='45', ax = fig.add_subplot(111))
    st.title("QQ-Plot")
    st.pyplot(fig)

# Function for KDE Estimation
def kde_estimation(data):
    fig = plt.figure()
    sns.kdeplot(data)
    st.title("Kernel Density Estimate")
    st.pyplot(fig)

# Function for Sharpe Ratio
def sharpe_ratio(returns, risk_free_rate=0):
    excess_returns = returns - risk_free_rate
    sharpe = excess_returns.mean() / excess_returns.std()
    st.write(f"Sharpe Ratio: {sharpe:.2f}")

# Function for Value at Risk (VaR)
def value_at_risk(data, confidence_level=0.95):
    var = np.percentile(data, (1 - confidence_level) * 100)
    st.write(f"Value at Risk (VaR) at {confidence_level*100}% confidence: {var:.2f}")

# Function for ACF (Autocorrelation Function)
def autocorrelation(data):
    fig, ax = plt.subplots()
    sm.graphics.tsa.plot_acf(data, lags=3, ax=ax)
    st.title("Autocorrelation Function (ACF)")
    st.pyplot(fig)

# Function for PACF (Partial Autocorrelation Function)
def partial_autocorrelation(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(data, lags=10, method='ywm', ax=ax)
    ax.set_title('PACF of Stock Returns')

    # Display the plot using Streamlit
    st.pyplot(fig)
def OptionPrice_FDM_Explicit(S0, K, r, sigma, T, Smax, m, n, option_type):
    #initialisation and parameters
    P  = np.zeros((m + 1, n + 1))
    S =  np.linspace(0, Smax, m + 1)
    dS = Smax / m
    dt= T / n
    ii = S / dS

    #boundary conditions
    if option_type == 'Call':
        P[m, :] = [Smax - K * np.exp(r * ( n - j) * dt) for j in range(n + 1)]
    if option_type == 'Put':
        P[0, :] = [K * np.exp(-r * ( n - j) * dt) for j in range(n + 1)]
    #terminal conditions
    if option_type == 'Call':
        P[:, n] = np.array(np.maximum(0, S - K))
    if option_type == 'Put':
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

def black_scholes(S, K, T, r, sigma, option_type):
    d1 = ((np.log(S / K) + (r + 0.5 * sigma ** 2) * T)) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "Call":
        return S * norm.cdf(d1) - (K * np.exp(-r * T) * norm.cdf(d2))
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
    if option_type == "Call":
        option_tree[:, N] = np.maximum(stock_tree[:, N] - K, 0)
    elif option_type == "Put":
        option_tree[:, N] = np.maximum(K - stock_tree[:, N], 0)

    # Backward induction to calculate option price
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = np.exp(-r * dt) * (p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1])
            if option_type == "Call":
                exercise_value = stock_tree[j, i] - K
            else:
                exercise_value = K - stock_tree[j, i]
            option_tree[j, i] = max(continuation_value, exercise_value)

    return round(option_tree[0, 0], 4)

def delta(S, K, T, r, sigma, option_type="Call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "Call":
            return norm.cdf(d1)
        elif option_type == "Put":
            return norm.cdf(d1) - 1
    except Exception as e:
        st.warning(f"Error calculating Delta: {e}")
        return None

def gamma(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))
    except Exception as e:
        st.warning(f"Error calculating Gamma: {e}")
        return None

def theta(S, K, T, r, sigma, option_type="call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == "call":
            return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)
    except Exception as e:
        st.warning(f"Error calculating Theta: {e}")
        return None

def vega(S, K, T, r, sigma):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return S * norm.pdf(d1) * np.sqrt(T)
    except Exception as e:
        st.warning(f"Error calculating Vega: {e}")
        return None

def rho(S, K, T, r, sigma, option_type="call"):
    try:
        d2 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)) - sigma * np.sqrt(T)
        if option_type == "call":
            return K * T * np.exp(-r * T) * norm.cdf(d2)
        elif option_type == "put":
            return -K * T * np.exp(-r * T) * norm.cdf(-d2)
    except Exception as e:
        st.warning(f"Error calculating Rho: {e}")
        return None


# Initialize session state for Paper Trading Portfolio
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {}

st.set_page_config(
    page_title="In The Money",
    page_icon="📈",
    layout="wide",  # Options are "centered" or "wide"
    initial_sidebar_state="expanded",  # "collapsed" or "expanded"
)

st.markdown("""
    <style>
        body {
            background-image: url('https://plus.unsplash.com/premium_photo-1672423154405-5fd922c11af2?w=600&auto=format&fit=crop&q=60&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxzZWFyY2h8MXx8Y29ycG9yYXRlJTIwYmFja2dyb3VuZHxlbnwwfHwwfHx8MA%3D%3D');
            background-size: cover;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation (Taskbar Menu)
st.sidebar.title("Stock Analysis App")
menu = st.sidebar.radio("Select a Section", ["Home", "Stock Prices", "Stock Options", "Financials", "Statistical Analysis", "Recommendations"])

# Sidebar: Select Stock
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL)", value="AAPL")

if ticker:
    stock = yf.Ticker(ticker)
    info = stock.info


    if menu == "Home":
        st.title("Welcome to the In The Money")
        st.write("""
        This app provides real-time stock analysis, including:
        - Live Stock Prices 
        - Options Chain Data 
        - Financial Statements 
        - Virtual Paper Trading 
        - AI-Based Stock Recommendations 
        """)


    elif menu == "Stock Prices":
        st.title(f"{ticker} Stock Price")
        price = stock.history(period="1d")["Close"].iloc[-1]
        st.metric("Current Price", f"${price:.2f}")
        st.write(f"**Sector:** {info.get('sector', 'N/A')}")
        st.write(f"**Industry:** {info.get('industry', 'N/A')}")


        period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"], index=3)
        interval = st.selectbox("Select Interval", ["1d", "5d", "1wk", "1mo", "3mo"], index=0)

        hist_data = stock.history(period=period, interval=interval)

        if not hist_data.empty:
            st.write("Price Chart")
            st.line_chart(hist_data['Close'])
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(label="Market Cap", value=f"${info.get('marketCap', 'N/A')/10000000000:.2f}B")
                st.metric(label="P/E Ratio", value=info.get("trailingPE", "N/A"))

            with col2:
                st.metric(label="52-Week High", value=f"${info.get('fiftyTwoWeekHigh', 'N/A')}")
                st.metric(label="52-Week Low", value=f"${info.get('fiftyTwoWeekLow', 'N/A')}")

            with col3:
                st.metric(label="Dividend Yield", value=f"{info.get('dividendYield', 'N/A')}")
                st.metric(label="Beta", value=info.get("beta", "N/A"))
            st.write("Historical Price Data")
            st.dataframe(hist_data[['Open', 'High', 'Low', 'Close', 'Volume']])



        else:
            st.warning("No historical data available.")







    elif menu == "Stock Options":
        Options_sub_menu = st.selectbox("Task",["Options Chain","Pricing"])

        if Options_sub_menu == "Options Chain":
            st.title(f"📜 {ticker} Options Chain")
            expiry_dates = stock.options
            selected_date = st.selectbox("Select Expiry Date", expiry_dates)

            if selected_date:
                options_chain = stock.option_chain(selected_date)
                st.write("Call Options")
                st.dataframe(options_chain.calls)
                st.write("Put Options")
                st.dataframe(options_chain.puts)
        else:
            S = st.number_input("Stock Price ($)", min_value=1.0, value=100.0, step=0.5)
            K = st.number_input("Strike Price ($)", min_value=1.0, value=100.0, step=0.5)
            T = st.number_input("Time to Expiration (years)", min_value=0.01, value=1.0, step=0.01)
            r = st.number_input("Risk-Free Rate (%)", min_value=0.0, value=5.0, step=0.1) / 100
            sigma = st.number_input("Volatility (%)", min_value=0.01, value=20.0, step=0.1) / 100
            N = st.slider("Number of Time Steps", min_value=10, max_value=500, value=100, step=10)
            option_type = st.selectbox("Option Type", ["Call", "Put"])

            Pricing_sub_menu = st.selectbox("Method",["Black Scholes","Finite Difference Method","Binomial Tree"])

            if Pricing_sub_menu == "Black Scholes":
                price = black_scholes(S,K,T,r,sigma,option_type)
            elif Pricing_sub_menu == "Finite Difference Method":
                S_max = st.number_input("Max Stock Price", min_value=1.0, value = 100.0, step =.5)
                price = OptionPrice_FDM_Explicit(S,K,r,sigma,T,S_max,N,N,option_type)
            elif Pricing_sub_menu == "Binomial Tree":
                price = binomial_tree(S,K,T,r,sigma,N,option_type)

            if st.button("Calculate Price"):
                st.success(f"💰 The {option_type} Option Price is: **${price}**")



    elif menu == "Financials":
        st.title(f"{ticker} Financial Statements")
        st.write("**Income Statement**")
        st.dataframe(stock.financials)
        st.write("**Balance Sheet**")
        st.dataframe(stock.balance_sheet)
        st.write("**Cash Flow Statement**")
        st.dataframe(stock.cashflow)

        url = f"https://finance.yahoo.com/quote/{ticker}/filings"
        st.markdown(f"📄 **[View Official SEC Filings (10-K Reports)]({url})**", unsafe_allow_html=True)

    elif menu =="Statistical Analysis":
        option = st.selectbox(
            "Choose a statistical test to run:",
            ["Select Test", "GARCH", "EGARCH", "QQ Plot", "KDE Estimation",
              "Value at Risk (VaR)", "ACF", "PACF"]
        )
        start_date = st.date_input("Start Date:")
        end_date = st.date_input("End Date")
        data = yf.download(ticker, start=start_date, end=end_date)
        stock_data = data['Close']
        returns = stock_data.pct_change().dropna()

        if st.button("Test"):
            if option == "GARCH":
                garch_model(returns)
            elif option == "EGARCH":
                egarch_model(returns)
            elif option == "QQ Plot":
                qq_plot(returns)
            elif option == "KDE Estimation":
                kde_estimation(returns)
            elif option == "Value at Risk (VaR)":
                value_at_risk(stock_data)
            elif option == "ACF":
                autocorrelation(returns)
            elif option == "PACF":
                partial_autocorrelation(returns)

    # 🤖 **Stock Recommendations**
    elif menu == "Recommendations":
        st.title("AI Stock Recommendation")
        pe_ratio = info.get("trailingPE", None)

        if pe_ratio:
            if pe_ratio < 15:
                st.success("✅ **Recommendation: Buy (Undervalued)**")
            elif 15 <= pe_ratio <= 25:
                st.warning("⚖️ **Recommendation: Hold (Fairly Valued)**")
            else:
                st.error("❌ **Recommendation: Sell (Overvalued)**")
        else:
            st.write("No P/E ratio available for recommendation.")


