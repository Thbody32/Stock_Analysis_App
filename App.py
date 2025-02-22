import streamlit as st
import yfinance as yf
import pandas as pd
import datetime
import numpy as np
from scipy.stats import norm
from arch import arch_model
from fpdf import FPDF
import os

# Black-Scholes pricing function
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d1)

# Function to add financial statements to PDF in a readable format
def add_financials_to_pdf(pdf, title, dataframe):
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, title, ln=True, align='L')
    pdf.ln(3)

    for index, row in dataframe.iterrows():
        row_text = f"{index}: " + " | ".join([f"{col}: {row[col]:,.2f}" if isinstance(row[col], (int, float)) else f"{col}: {row[col]}" for col in dataframe.columns])
        pdf.multi_cell(0, 5, row_text)
        pdf.ln(1)

    pdf.ln(5)

# Streamlit UI
st.set_page_config(page_title="Options Pricing & Analysis App", layout="wide")
st.title("Options Pricing & Analysis App")

# Sidebar Menu
menu_selection = st.sidebar.radio("Navigation", ["Stock Data", "Options Chain", "Implied Volatility", "Options Pricing", "GARCH Model", "Equity Report", "Paper Trading"])

# User input for stock symbol
symbol = st.text_input("Enter Stock Symbol:", value="AAPL").upper()

time_period = st.selectbox("Select Time Period:", ["1wk", "1mo", "6mo", "1y", "5y", "max"], index=3)

if symbol:
    stock = yf.Ticker(symbol)
    stock_data = stock.history(period=time_period)

    if menu_selection == "Stock Data":
        st.subheader(f"Stock Data for {symbol} ({time_period})")
        st.line_chart(stock_data["Close"])

    elif menu_selection == "Equity Report":
        st.subheader("Full Equity Report")
        company_overview = stock.info.get("longBusinessSummary", "No summary available.")

        income_statement = stock.financials
        balance_sheet = stock.balance_sheet
        cash_flow = stock.cashflow

        pe_ratio = stock.info.get('trailingPE', 'N/A')
        pb_ratio = stock.info.get('priceToBook', 'N/A')
        debt_equity = stock.info.get('debtToEquity', 'N/A')
        roe = stock.info.get('returnOnEquity', 'N/A')

        # Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, f"Equity Report: {symbol}", ln=True, align='C')
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Company Overview:\n{company_overview}")
        pdf.ln(5)
        pdf.cell(0, 10, "Financial Ratios:", ln=True)
        pdf.cell(0, 10, f"P/E Ratio: {pe_ratio}", ln=True)
        pdf.cell(0, 10, f"P/B Ratio: {pb_ratio}", ln=True)
        pdf.cell(0, 10, f"Debt/Equity Ratio: {debt_equity}", ln=True)
        pdf.cell(0, 10, f"ROE: {roe}", ln=True)

        pdf.ln(5)
        add_financials_to_pdf(pdf, "Income Statement", income_statement)
        add_financials_to_pdf(pdf, "Balance Sheet", balance_sheet)
        add_financials_to_pdf(pdf, "Cash Flow Statement", cash_flow)

        pdf_filename = f"{symbol}_Equity_Report.pdf"
        pdf.output(pdf_filename)

        with open(pdf_filename, "rb") as file:
            st.download_button("Download Equity Report as PDF", data=file, file_name=pdf_filename, mime="application/pdf")

        os.remove(pdf_filename)

    elif menu_selection == "Paper Trading":
        st.subheader("Paper Trading")
        initial_balance = st.number_input("Enter Initial Balance:", min_value=100.0, value=10000.0)
        if "portfolio" not in st.session_state:
            st.session_state.portfolio = {"cash": initial_balance, "holdings": {}}

        action = st.radio("Select Action:", ["Buy", "Sell"])
        quantity = st.number_input("Enter Quantity:", min_value=1, value=10)
        current_price = stock.history(period="1d")["Close"].iloc[-1]

        if st.button("Execute Trade"):
            if action == "Buy":
                cost = quantity * current_price
                if cost <= st.session_state.portfolio["cash"]:
                    st.session_state.portfolio["cash"] -= cost
                    if symbol in st.session_state.portfolio["holdings"]:
                        st.session_state.portfolio["holdings"][symbol] += quantity
                        st.session_state.portfolio["Time"][symbol] += datetime.now()
                    else:
                        st.session_state.portfolio["holdings"][symbol] = quantity
                    st.success(f"Bought {quantity} shares of {symbol} at ${current_price:.2f} each.")
                else:
                    st.error("Insufficient funds!")
            else:  # Sell
                if symbol in st.session_state.portfolio["holdings"] and st.session_state.portfolio["holdings"][symbol] >= quantity:
                    st.session_state.portfolio["holdings"][symbol] -= quantity
                    st.session_state.portfolio["cash"] += quantity * current_price
                    st.success(f"Sold {quantity} shares of {symbol} at ${current_price:.2f} each.")
                else:
                    st.error("Not enough shares to sell!")

        st.subheader("Portfolio Summary")
        st.write(f"Cash: ${st.session_state.portfolio['cash']:.2f}")
        st.write("Holdings:")
        st.write(st.session_state.portfolio["holdings"]["time"])

