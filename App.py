import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
from Options_Pricing_Methods import black_scholes, binomial_tree, OptionPrice_FDM_Explicit
from Statistical_Tests import garch_model, egarch_model, kde_estimation, value_at_risk, autocorrelation, partial_autocorrelation, sharpe_ratio, qq_plot

# Greeks calculation functions

def delta(S, K, T, r, sigma, option_type="call"):
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == "call":
            return norm.cdf(d1)
        elif option_type == "put":
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
                price = binomial_tree(S, K, T, r, sigma, N, option_type.lower())
                st.success(f"💰 The {option_type} Option Price is: **${price}**")

                greek_query = st.selectbox("Greeks and implied volatility?",["Yes","No"])

                if greek_query == "Yes":
                    st.subheader("Option Greeks:")

                    col1, col2, col3, col4, col5 = st.columns(5)
                    with col1:
                        st.metric(label = "Delta",value = f"{delta(S, K, T, r, sigma, option_type):.2f}")
                    with col2:
                        st.metric(label ="Gamma",value = f"{gamma(S, K, T, r, sigma):.2f}")
                    with col3:
                        st.metric(label = "Theta",value= f"{theta(S, K, T, r, sigma, option_type):.2f}")
                    with col4:
                        st.metric(label = "Vega",value= f"{vega(S, K, T, r, sigma):.2f}")
                    with col5:
                        st.metric(label = "Rho",value = f"{rho(S, K, T, r, sigma, option_type):.2f}")
                elif greek_query == "No":
                    st.write("Ok no problem")





    elif menu == "Financials":
        st.title(f"{ticker} Financial Statements")
        st.write("**Income Statement**")
        st.dataframe(stock.financials)
        st.write("**Balance Sheet**")
        st.dataframe(stock.balance_sheet)
        st.write("**Cash Flow Statement**")
        st.dataframe(stock.cashflow)

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


    elif menu == "Paper Trading":
        st.title("Paper Trading Simulator")
        action = st.radio("Buy or Sell", ["Buy", "Sell"])
        shares = st.number_input("Enter Number of Shares", min_value=1, value=10)

        if st.button("Execute Trade"):
            price = stock.history(period="1d")["Close"].iloc[-1]
            total_cost = price * shares

            if action == "Buy":
                st.session_state.portfolio[ticker] = st.session_state.portfolio.get(ticker, 0) + shares
                st.success(f"✅ Bought {shares} shares of {ticker} at ${price:.2f}")
            elif action == "Sell":
                if ticker in st.session_state.portfolio and st.session_state.portfolio[ticker] >= shares:
                    st.session_state.portfolio[ticker] -= shares
                    st.success(f"✅ Sold {shares} shares of {ticker} at ${price:.2f}")
                else:
                    st.error("❌ Not enough shares to sell!")

        st.write("Your Portfolio:", st.session_state.portfolio)

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


