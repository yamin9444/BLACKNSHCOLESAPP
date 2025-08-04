#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np
import yfinance as yf
import pandas as pd

# === Fonctions Black-Scholes ===
def blackScholes(S, K, r, T, sigma, type="c"):
    d1 = (np.log(S/K) + (r + sigma**2/2)* T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "c":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif type == "p":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

def implied_volatility(market_price, S, K, r, T, type='c'):
    func = lambda sigma: blackScholes(S, K, r, T, sigma, type) - market_price
    try:
        iv = brentq(func, 1e-5, 5)
        return iv
    except ValueError:
        return np.nan

# === Fonctions Greeks ===
def black_scholes_greeks(S, K, r, T, sigma, type="c"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if type == "c":
        delta = norm.cdf(d1)
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    return {"Delta": delta, "Gamma": gamma, "Vega": vega, "Theta": theta, "Rho": rho}

# === Sidebar (ticker & récupération spot) ===
st.title("Application Black-Scholes Pricing")

st.sidebar.header("Entrée automatique du prix spot")
ticker = st.sidebar.text_input("Ticker Yahoo Finance (ex: AAPL)", value="AAPL")
btn = st.sidebar.button("Récupérer le prix spot")
spot_price = None

if btn:
    try:
        ticker_obj = yf.Ticker(ticker)
        spot_price = ticker_obj.history(period="1d")["Close"].iloc[-1]
        st.sidebar.success(f"Dernier prix {ticker.upper()} : {spot_price:.2f} $")
    except Exception as e:
        st.sidebar.error(f"Erreur récupération : {e}")

st.write("#### Remplis les champs (exemples à droite)")

# === Inputs principaux avec exemples ===
col1, col2 = st.columns([2,1])

with col1:
    S = st.number_input("Spot (S) – dernier prix de l’action", value=spot_price if spot_price else 100.0)
    K = st.number_input("Strike (K) – prix d’exercice", value=100.0)
    r = st.number_input("Taux sans risque (r)", value=0.03, format="%.6f", help="Exemple : si taux 4,49% alors entre 0.0449")
    T_days = st.number_input("Maturité (en jours)", value=30, help="Exemple : 30 jours = 0.082 en années")
    sigma = st.number_input("Volatilité (sigma, en % annualisée)", value=25.0, help="Exemple : 25% = 0.25")
    type_opt = st.selectbox("Type d’option", ("Call", "Put"))
    type_clean = "c" if type_opt == "Call" else "p"
    market_price = st.number_input("Prix de l'option observé (pour calcul IV)", value=0.0, help="Laisse à zéro si tu veux juste le prix Black-Scholes")

with col2:
    st.write("**Exemples**")
    st.markdown("- Spot = 190.12")
    st.markdown("- Strike = 200")
    st.markdown("- Taux sans risque = 0.0449 *(si taux 4,49%)*")
    st.markdown("- Maturité : 30 jours = 0.082 *(= 30/365)*")
    st.markdown("- Volatilité : 25% = 0.25")
    st.markdown("- Observé marché : 4.50")

T = T_days / 365
sigma_val = sigma / 100

# === Calcul prix, IV, Greeks ===
call_price = blackScholes(S, K, r, T, sigma_val, type_clean)
st.success(f"**Prix Black-Scholes : {call_price:.2f} $**")

if market_price > 0:
    iv = implied_volatility(market_price, S, K, r, T, type_clean)
    st.info(f"**Volatilité implicite (IV) annualisée : {iv:.2%}**")
    # IV transformé en différentes périodicités
    ivs = {
        "Annualisée": iv,
        "Quarterly": iv * np.sqrt(1/4),
        "Mensuelle": iv * np.sqrt(1/12),
        "Hebdo": iv * np.sqrt(1/52),
        "Daily": iv * np.sqrt(1/252)
    }
    iv_df = pd.DataFrame({
        "Horizon": ["Annualisée", "Quarter", "Mois", "Semaine", "Jour"],
        "IV (%)": [f"{ivs['Annualisée']*100:.2f}", f"{ivs['Quarterly']*100:.2f}", f"{ivs['Mensuelle']*100:.2f}", f"{ivs['Hebdo']*100:.2f}", f"{ivs['Daily']*100:.2f}"]
    })
    st.write("#### IV convertie pour chaque période")
    st.table(iv_df)
else:
    iv = None

# === Affichage des Greeks ===
if T > 0 and sigma > 0:
    greeks = black_scholes_greeks(S, K, r, T, sigma_val, type_clean)
    st.write("### Greeks")
    st.write(pd.DataFrame(greeks, index=["Valeur"]).T)
else:
    st.warning("Renseigne tous les paramètres pour obtenir les Greeks.")

# === Design / tips / astuces UX ===
st.caption("Astuce : le spot peut être rempli automatiquement via Yahoo Finance dans la barre latérale (ticker US, ex: AAPL, MSFT, TSLA).")

st.markdown("""---  
*Made by DeclicFinance – Personnalise ce template pour ton usage professionnel (logo, disclaimer, couleurs…)*  
""")


