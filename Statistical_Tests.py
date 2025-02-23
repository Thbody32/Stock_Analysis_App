import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from arch import arch_model
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_pacf


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