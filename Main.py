import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import norm
import math

st.set_page_config(
    page_title="Black-Scholes Options Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    .call-option {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .put-option {
        background: linear-gradient(135deg, #f44336 0%, #da190b 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .greek-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def black_scholes_price(S, K, T, r, sigma, q=0, option_type='call'):
    if T <= 0 or sigma <= 0:
        return 0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == 'call':
        price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

    return max(0, price)


def calculate_greeks(S, K, T, r, sigma, q=0):
    if T <= 0 or sigma <= 0:
        return {
            'delta_call': 0, 'delta_put': 0, 'gamma': 0, 'vega': 0,
            'theta_call': 0, 'theta_put': 0, 'rho_call': 0, 'rho_put': 0,
            'd1': 0, 'd2': 0
        }

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta_call = np.exp(-q * T) * norm.cdf(d1)
    delta_put = np.exp(-q * T) * (norm.cdf(d1) - 1)

    gamma = np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100

    theta_call = (-(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1)) / 365

    theta_put = (-(S * norm.pdf(d1) * sigma * np.exp(-q * T)) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1)) / 365

    rho_call = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    rho_put = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    return {
        'delta_call': delta_call, 'delta_put': delta_put,
        'gamma': gamma, 'vega': vega,
        'theta_call': theta_call, 'theta_put': theta_put,
        'rho_call': rho_call, 'rho_put': rho_put,
        'd1': d1, 'd2': d2
    }


st.markdown('<h1 class="main-header">Black-Scholes Options Calculator</h1>', unsafe_allow_html=True)

st.sidebar.header("Option Parameters")

S = st.sidebar.number_input("Spot Price ($)", value=100.0, min_value=0.01, step=0.01, format="%.2f")
K = st.sidebar.number_input("Strike Price ($)", value=100.0, min_value=0.01, step=0.01, format="%.2f")
T = st.sidebar.number_input("Time to Expiry (years)", value=1.0, min_value=0.001, step=0.01, format="%.3f")
r = st.sidebar.number_input("Risk-Free Rate (%)", value=5.0, min_value=0.0, step=0.1, format="%.1f") / 100
sigma = st.sidebar.number_input("Volatility (%)", value=20.0, min_value=0.1, step=0.1, format="%.1f") / 100
q = st.sidebar.number_input("Dividend Yield (%)", value=0.0, min_value=0.0, step=0.1, format="%.1f") / 100

call_price = black_scholes_price(S, K, T, r, sigma, q, 'call')
put_price = black_scholes_price(S, K, T, r, sigma, q, 'put')
greeks = calculate_greeks(S, K, T, r, sigma, q)

call_intrinsic = max(0, S - K)
put_intrinsic = max(0, K - S)

tab1, tab2, tab3 = st.tabs(["Calculator", "Price Sensitivity", "Greeks Analysis"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
        <div class="call-option">
            <h3>Call Option</h3>
            <h2>${call_price:.2f}</h2>
            <p>Intrinsic Value: ${call_intrinsic:.2f}</p>
            <p>Time Value: ${call_price - call_intrinsic:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="put-option">
            <h3>Put Option</h3>
            <h2>${put_price:.2f}</h2>
            <p>Intrinsic Value: ${put_intrinsic:.2f}</p>
            <p>Time Value: ${put_price - put_intrinsic:.2f}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("d₁", f"{greeks['d1']:.4f}")
    with col2:
        st.metric("d₂", f"{greeks['d2']:.4f}")
    with col3:
        st.metric("Moneyness", f"{S / K:.4f}")
    with col4:
        st.metric("Time to Expiry", f"{T:.3f} years")

with tab2:
    st.subheader("Price Sensitivity Analysis")

    spot_range = np.linspace(S * 0.6, S * 1.4, 50)
    call_prices = []
    put_prices = []
    call_intrinsics = []
    put_intrinsics = []

    for spot in spot_range:
        call_prices.append(black_scholes_price(spot, K, T, r, sigma, q, 'call'))
        put_prices.append(black_scholes_price(spot, K, T, r, sigma, q, 'put'))
        call_intrinsics.append(max(0, spot - K))
        put_intrinsics.append(max(0, K - spot))

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=spot_range, y=call_prices, mode='lines',
                             name='Call Price', line=dict(color='green', width=3)))
    fig.add_trace(go.Scatter(x=spot_range, y=put_prices, mode='lines',
                             name='Put Price', line=dict(color='red', width=3)))

    fig.add_trace(go.Scatter(x=spot_range, y=call_intrinsics, mode='lines',
                             name='Call Intrinsic', line=dict(color='green', width=1, dash='dash')))
    fig.add_trace(go.Scatter(x=spot_range, y=put_intrinsics, mode='lines',
                             name='Put Intrinsic', line=dict(color='red', width=1, dash='dash')))

    fig.add_vline(x=S, line_dash="dot", line_color="blue",
                  annotation_text=f"Current Spot: ${S}")

    fig.add_vline(x=K, line_dash="dot", line_color="purple",
                  annotation_text=f"Strike: ${K}")

    fig.update_layout(
        title="Option Prices vs Spot Price",
        xaxis_title="Spot Price ($)",
        yaxis_title="Option Price ($)",
        hovermode='x unified',
        template='plotly_white'
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Volatility Sensitivity")

    vol_range = np.linspace(0.1, 1.0, 30)
    call_vol_prices = []
    put_vol_prices = []

    for vol in vol_range:
        call_vol_prices.append(black_scholes_price(S, K, T, r, vol, q, 'call'))
        put_vol_prices.append(black_scholes_price(S, K, T, r, vol, q, 'put'))

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(x=vol_range * 100, y=call_vol_prices, mode='lines',
                                 name='Call Price', line=dict(color='green', width=3)))
    fig_vol.add_trace(go.Scatter(x=vol_range * 100, y=put_vol_prices, mode='lines',
                                 name='Put Price', line=dict(color='red', width=3)))

    fig_vol.add_vline(x=sigma * 100, line_dash="dot", line_color="blue",
                      annotation_text=f"Current Vol: {sigma * 100:.1f}%")

    fig_vol.update_layout(
        title="Option Prices vs Volatility",
        xaxis_title="Volatility (%)",
        yaxis_title="Option Price ($)",
        template='plotly_white'
    )

    st.plotly_chart(fig_vol, use_container_width=True)

with tab3:
    st.subheader("The Greeks")

    greeks_data = [
        ("Delta", "Price sensitivity to underlying asset",
         greeks['delta_call'], greeks['delta_put'], "4f"),
        ("Gamma", "Rate of change of delta",
         greeks['gamma'], greeks['gamma'], "4f"),
        ("Vega", "Sensitivity to volatility (per 1%)",
         greeks['vega'], greeks['vega'], "4f"),
        ("Theta", "Time decay (per day)",
         greeks['theta_call'], greeks['theta_put'], "4f"),
        ("Rho", "Interest rate sensitivity (per 1%)",
         greeks['rho_call'], greeks['rho_put'], "4f")
    ]

    for greek_name, description, call_val, put_val, fmt in greeks_data:
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            st.markdown(f"""
            <div class="greek-card">
                <h4>{greek_name}</h4>
                <p style="margin: 0; color: #666;">{description}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.metric("Call", f"{call_val:.{fmt[-2:]}}")

        with col3:
            st.metric("Put", f"{put_val:.{fmt[-2:]}}")

    st.subheader("Greeks Visualization")

    spot_range_greeks = np.linspace(S * 0.8, S * 1.2, 30)
    deltas_call = []
    deltas_put = []
    gammas = []

    for spot in spot_range_greeks:
        greeks_temp = calculate_greeks(spot, K, T, r, sigma, q)
        deltas_call.append(greeks_temp['delta_call'])
        deltas_put.append(greeks_temp['delta_put'])
        gammas.append(greeks_temp['gamma'])

    fig_greeks = go.Figure()
    fig_greeks.add_trace(go.Scatter(x=spot_range_greeks, y=deltas_call, mode='lines',
                                    name='Call Delta', line=dict(color='green')))
    fig_greeks.add_trace(go.Scatter(x=spot_range_greeks, y=deltas_put, mode='lines',
                                    name='Put Delta', line=dict(color='red')))

    fig_greeks.add_vline(x=S, line_dash="dot", line_color="blue",
                         annotation_text=f"Current Spot: ${S}")

    fig_greeks.update_layout(
        title="Delta vs Spot Price",
        xaxis_title="Spot Price ($)",
        yaxis_title="Delta",
        template='plotly_white'
    )

    st.plotly_chart(fig_greeks, use_container_width=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>This calculator uses  Black-Scholes-Merton model for European options pricing</p>
</div>
""", unsafe_allow_html=True)
