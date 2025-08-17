import streamlit as st
import numpy as np
import numpy_financial as npf
import pandas as pd
import plotly.graph_objects as go
from fpdf import FPDF
import base64
from datetime import datetime
import requests
import tempfile
import os
import matplotlib.pyplot as plt

# ------------- Constants & Defaults -------------

# Base tax thresholds (unchanged)
TAX_THRESHOLDS_BASE = {
    "free_zone_exemption": 375000,  # AED
    "small_business_relief": 375000,  # AED
    "large_mne_revenue": 750000000   # EUR
}

# Static fallback rates (used if live fetch fails)
EXCHANGE_RATES_FALLBACK = {
    "AED": {"USD": 0.272, "EUR": 0.25},
    "USD": {"AED": 3.67, "EUR": 0.92},
    "EUR": {"AED": 4.0, "USD": 1.09}
}

# Initialize the mutable exchange rates dict (will be updated by fetch)
EXCHANGE_RATES = EXCHANGE_RATES_FALLBACK.copy()

# Risks and mitigations
RISK_MITIGATIONS = {
    "Market Competition": "Conduct thorough market research and develop a unique value proposition.",
    "Regulatory Changes": "Engage local legal experts to stay compliant with evolving regulations.",
    "Economic Volatility": "Diversify revenue streams and maintain cash reserves.",
    "Supply Chain Disruption": "Develop relationships with multiple suppliers.",
    "Talent Acquisition": "Invest in competitive compensation and training programs.",
    "Currency Fluctuation": "Use hedging strategies to mitigate currency risks.",
    "Operational Challenges": "Implement robust operational processes and monitoring.",
    "Cultural Adaptation": "Hire local experts to navigate cultural nuances.",
    "Local Market Knowledge": "Partner with local firms to gain market insights.",
    "Legal Compliance": "Regularly review compliance with local laws.",
    "Technology Adoption": "Invest in scalable and adaptable technology solutions.",
    "Customer Acquisition Costs": "Optimize marketing strategies to reduce acquisition costs."
}

# Monetary keys for conversion
MONETARY_KEYS = ["initial_investment", "projected_revenue", "fixed_costs", "capex"]

# Defaults
DEFAULTS = {
    "currency": "AED",
    "initial_investment": 500000.0,
    "project_lifespan": 5,
    "discount_rate": 10.0,
    "free_zone": True,
    "meets_substance": True,
    "non_qual_percent": 0.0,
    "is_large_mne": False,
    "projected_revenue": 750000.0,
    "revenue_growth": 12.0,
    "operating_margin": 25.0,
    "fixed_costs": 150000.0,
    "inflation_rate": 2.0,
    "variable_cost_percent": 35.0,
    "capex": 0.0
}

# Initialize session state
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ------------- Page config & CSS -------------

st.set_page_config(page_title="UAE Business Expansion Simulator", layout="centered", page_icon="📊")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; color: #2c3e50; }
    .stNumberInput, .stSelectbox, .stSlider { background-color: white; border-radius: 6px; padding: 8px; border: 1px solid #dee2e6; }
    .stButton>button { background-color: #2c3e50; color: white; border-radius: 6px; padding: 8px 16px; font-weight: 500; transition: all 0.3s; }
    .stButton>button:hover { background-color: #1a252f; transform: translateY(-1px); }
    .st-bb { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.05); }
    .header { color: #2c3e50; font-weight: 600; }
    .metric-card { background-color: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 10px; }
    .risk-item { background-color: #f8d7da; padding: 5px 10px; border-radius: 4px; margin: 3px 0; font-size: 14px; }
    .stButton>button:focus { outline: 2px solid #007bff; outline-offset: 2px; }
    .metric-card h4 { color: #2c3e50; margin-bottom: 8px; }
    .high-contrast { color: #000000; background-color: #ffffff; }
    .stSelectbox > div > div > select { font-size: 16px; color: #2c3e50; background-color: white; padding: 8px; border: 1px solid #2c3e50; border-radius: 4px; appearance: auto; }
    .stSelectbox label { font-size: 16px; font-weight: 600; color: #2c3e50; }
    </style>
    """, unsafe_allow_html=True)

# ------------- Title -------------

st.title("📈 UAE Business Expansion Simulator")
st.markdown("""
    <p class="header">Comprehensive financial analysis tool for evaluating business expansion in the UAE market</p>
    <p style="color: #6c757d;">Use the sidebar to input parameters and explore financial projections.</p>
    """, unsafe_allow_html=True)

# ------------- Live Exchange Rates -------------

@st.cache_data(ttl=60*60*12)  # refresh every 12 hours
def fetch_exchange_rates():
    """
    Fetches AED, USD, EUR cross rates using a public API and returns a nested dict like EXCHANGE_RATES.
    Falls back to static rates on any error.
    """
    try:
        # Use one reliable free source (no key): open.er-api.com
        # Get base AED first
        r_aed = requests.get("https://open.er-api.com/v6/latest/AED", timeout=6)
        r_usd = requests.get("https://open.er-api.com/v6/latest/USD", timeout=6)
        r_eur = requests.get("https://open.er-api.com/v6/latest/EUR", timeout=6)
        if r_aed.status_code == 200 and r_usd.status_code == 200 and r_eur.status_code == 200:
            aed_rates = r_aed.json().get("rates", {})
            usd_rates = r_usd.json().get("rates", {})
            eur_rates = r_eur.json().get("rates", {})
            # Build cross table (only the pairs we need)
            out = {
                "AED": {"USD": float(aed_rates.get("USD", EXCHANGE_RATES_FALLBACK["AED"]["USD"])),
                        "EUR": float(aed_rates.get("EUR", EXCHANGE_RATES_FALLBACK["AED"]["EUR"]))},
                "USD": {"AED": float(usd_rates.get("AED", EXCHANGE_RATES_FALLBACK["USD"]["AED"])),
                        "EUR": float(usd_rates.get("EUR", EXCHANGE_RATES_FALLBACK["USD"]["EUR"]))},
                "EUR": {"AED": float(eur_rates.get("AED", EXCHANGE_RATES_FALLBACK["EUR"]["AED"])),
                        "USD": float(eur_rates.get("USD", EXCHANGE_RATES_FALLBACK["EUR"]["USD"]))},
            }
            return out
    except Exception:
        pass
    return EXCHANGE_RATES_FALLBACK

# Refresh global rates
EXCHANGE_RATES = fetch_exchange_rates()

# ------------- Helpers -------------

def convert_currency(value, from_curr, to_curr):
    if from_curr == to_curr:
        return value
    return value * EXCHANGE_RATES[from_curr][to_curr]

def fmt_curr(value, currency_symbol):
    if value is None:
        return "N/A"
    return f"{currency_symbol} {value:,.0f}"

def fmt_pct(value, decimals=1):
    return f"{value:.{decimals}f}%" if value is not None else "N/A"

def fmt_yrs(value):
    return f"{value:.1f} years" if value is not None else "N/A"

def get_currency_symbol(currency):
    return "AED" if currency == "AED" else "$"

def get_currency_symbol_pdf(currency):
    return "AED" if currency == "AED" else "$"

def currency_selector():
    """Displays the currency selector and forces app reload when currency changes."""
    with st.sidebar:
        st.header("Currency Selection")
        currency_options = ["AED", "USD"]
        if "currency" not in st.session_state:
            st.session_state.currency = "AED"
        try:
            current_index = currency_options.index(st.session_state.currency)
        except ValueError:
            current_index = 0
        new_currency = st.selectbox(
            "Select Currency",
            currency_options,
            index=current_index,
            key="currency_widget",
            help="Select the currency for all financial calculations"
        )
        if new_currency != st.session_state.currency:
            st.session_state.currency = new_currency
            st.rerun()

def get_tax_thresholds(currency):
    thresholds = {}
    thresholds["free_zone_exemption"] = convert_currency(
        TAX_THRESHOLDS_BASE["free_zone_exemption"], "AED", currency
    )
    thresholds["small_business_relief"] = convert_currency(
        TAX_THRESHOLDS_BASE["small_business_relief"], "AED", currency
    )
    thresholds["large_mne_revenue"] = convert_currency(
        TAX_THRESHOLDS_BASE["large_mne_revenue"], "EUR", currency
    )
    return thresholds

def get_user_inputs():
    """Collects all user inputs using direct key-based state management."""
    currency_selector()
    tax_thresholds = get_tax_thresholds(st.session_state.currency)

    with st.sidebar:
        st.header("📋 Business Parameters")
        with st.expander("💰 Investment Details", expanded=True):
            st.number_input(
                "Initial Investment:",
                min_value=10000.0,  # keep your safety min
                step=10000.0,
                help="Total capital required for expansion setup",
                key="initial_investment"
            )
            st.slider("Project Lifespan (years):", 1, 15, help="Duration of the expansion project", key="project_lifespan")
            st.slider("Discount Rate (%):", 0.1, 30.0, step=0.1, help="Required rate of return or cost of capital", key="discount_rate")

        with st.expander("🇦🇪 UAE Business Environment", expanded=True):
            st.checkbox("Registered in Free Zone", help="Operating within UAE-designated Free Zone area", key="free_zone")
            st.checkbox("Meets Economic Substance Requirements", help="Business has significant physical presence in UAE", key="meets_substance")
            st.slider("Non-Qualifying Income (% of Revenue):", 0.0, 100.0, step=1.0, help="Percentage of revenue subject to UAE corporate tax", key="non_qual_percent")
            st.checkbox("Large Multinational (Revenue ≥ €750M)", help="Part of a multinational group with global revenue ≥ €750M", key="is_large_mne")

        with st.expander("📈 Financial Projections", expanded=True):
            st.number_input("Annual Revenue:", min_value=1000.0, step=50000.0, help="Base year revenue projection", key="projected_revenue")
            st.slider("Annual Revenue Growth (%):", 0.0, 50.0, step=0.5, help="Expected annual revenue growth rate", key="revenue_growth")
            st.slider("Operating Margin (%):", 1.0, 50.0, step=0.5, help="Operating income as a percentage of revenue", key="operating_margin")
            st.number_input("Annual Fixed Costs:", min_value=1000.0, step=10000.0, help="Fixed operating costs per year", key="fixed_costs")
            st.slider("Fixed Cost Inflation Rate (%):", 0.0, 10.0, step=0.1, help="Annual inflation rate for fixed costs", key="inflation_rate")
            st.slider("Variable Costs (% of Revenue):", 0.0, 95.0, step=1.0, help="Variable costs as a percentage of revenue", key="variable_cost_percent")
            st.number_input("Annual Capital Expenditure:", min_value=0.0, step=10000.0, help="Annual capital expenditures (e.g., equipment)", key="capex")

    return {
        "initial_investment": st.session_state.initial_investment,
        "project_lifespan": st.session_state.project_lifespan,
        "discount_rate": st.session_state.discount_rate,
        "free_zone": st.session_state.free_zone,
        "meets_substance": st.session_state.meets_substance,
        "non_qual_percent": st.session_state.non_qual_percent,
        "is_large_mne": st.session_state.is_large_mne,
        "projected_revenue": st.session_state.projected_revenue,
        "revenue_growth": st.session_state.revenue_growth,
        "operating_margin": st.session_state.operating_margin,
        "fixed_costs": st.session_state.fixed_costs,
        "inflation_rate": st.session_state.inflation_rate,
        "variable_cost_percent": st.session_state.variable_cost_percent,
        "capex": st.session_state.capex,
        "currency": st.session_state.currency,
        "tax_thresholds": tax_thresholds
    }

@st.cache_data
def validate_inputs(inputs):
    errors = []
    if inputs["initial_investment"] <= 0:
        errors.append("Initial investment must be greater than zero.")
    if inputs["projected_revenue"] <= 0:
        errors.append("Annual revenue must be greater than zero.")
    if inputs["operating_margin"] <= 0:
        errors.append("Operating margin must be positive.")
    if inputs["variable_cost_percent"] >= 100:
        errors.append("Variable costs must be less than 100% of revenue.")
    if inputs["discount_rate"] <= 0:
        errors.append("Discount rate must be positive.")
    if inputs["projected_revenue"] > 0 and inputs["initial_investment"] > inputs["projected_revenue"] * 10:
        errors.append("Initial investment seems unusually high compared to projected revenue.")
    if inputs["projected_revenue"] > 0:
        total_cost_ratio = inputs["variable_cost_percent"] + (inputs["fixed_costs"] / inputs["projected_revenue"] * 100)
        if total_cost_ratio >= inputs["operating_margin"] + inputs["variable_cost_percent"]:
            errors.append("Total costs appear to exceed reasonable levels relative to revenue and operating margin.")
    if errors:
        return False, errors
    warnings = []
    if inputs["projected_revenue"] > 0 and inputs["fixed_costs"] > inputs["projected_revenue"] * 0.5:
        warnings.append("Fixed costs seem high relative to projected revenue.")
    if inputs["revenue_growth"] > 30:
        warnings.append("Revenue growth rate seems unusually high. Please verify.")
    return True, warnings

@st.cache_data
def calculate_cash_flows(inputs):
    years = np.arange(1, inputs["project_lifespan"] + 1)
    revenues = np.zeros_like(years, dtype=float)
    cash_flows = np.zeros_like(years, dtype=float)
    fixed_costs_yearly = np.zeros_like(years, dtype=float)
    for i, year in enumerate(years):
        revenues[i] = inputs["projected_revenue"] * (1 + inputs["revenue_growth"] / 100) ** (year - 1)
        fixed_costs_yearly[i] = inputs["fixed_costs"] * (1 + inputs["inflation_rate"] / 100) ** (year - 1)
        variable_costs = revenues[i] * (inputs["variable_cost_percent"] / 100)
        operating_income = revenues[i] * (inputs["operating_margin"] / 100)
        cash_flows[i] = operating_income - fixed_costs_yearly[i] - inputs["capex"]
    return years, revenues, cash_flows, fixed_costs_yearly

@st.cache_data
def calculate_taxes(inputs, revenues, cash_flows):
    non_qual_incomes = revenues * (inputs["non_qual_percent"] / 100)
    thresholds = inputs["tax_thresholds"]
    def get_tax_rate(revenue, non_qual_income, cash_flow):
        taxable_income = max(0, cash_flow)
        if inputs["free_zone"] and inputs["meets_substance"]:
            if non_qual_income <= thresholds["free_zone_exemption"]:
                return 0.0
            else:
                taxable_portion = non_qual_income - thresholds["free_zone_exemption"]
                return (taxable_portion / revenue) * 9.0 if revenue > 0 else 0.0
        elif taxable_income <= thresholds["small_business_relief"]:
            return 0.0
        elif inputs["is_large_mne"]:
            return 15.0
        else:
            return 9.0
    tax_rates = np.array([get_tax_rate(rev, nqi, cf) for rev, nqi, cf in zip(revenues, non_qual_incomes, cash_flows)])
    after_tax_flows = cash_flows * (1 - tax_rates / 100)
    return tax_rates, after_tax_flows

@st.cache_data
def calculate_financial_metrics(inputs, after_tax_flows):
    discount_factors = 1 / (1 + inputs["discount_rate"] / 100) ** np.arange(1, inputs["project_lifespan"] + 1)
    discounted_flows = after_tax_flows * discount_factors
    npv = np.sum(discounted_flows) - inputs["initial_investment"]
    cash_flow_series = np.concatenate(([-inputs["initial_investment"]], after_tax_flows))
    try:
        irr = npf.irr(cash_flow_series) * 100
        irr_error = None
    except Exception:
        irr = None
        irr_error = "Cannot calculate IRR due to consistent cash flow signs. Please review projections."
    cumulative_cash_flow = np.cumsum(after_tax_flows)
    payback_period = None
    if after_tax_flows.any() and after_tax_flows.max() > 0:
        for i, cum_flow in enumerate(cumulative_cash_flow):
            if cum_flow >= inputs["initial_investment"]:
                if i == 0:
                    payback_period = 1 if inputs["initial_investment"] <= 0 else (inputs["initial_investment"] / after_tax_flows[i])
                else:
                    partial_year = (inputs["initial_investment"] - cumulative_cash_flow[i-1]) / after_tax_flows[i]
                    payback_period = i + partial_year
                break
    pv = np.sum(discounted_flows)
    profitability_index = pv / inputs["initial_investment"] if inputs["initial_investment"] > 0 else None
    contribution_margin = 1 - inputs["variable_cost_percent"] / 100
    break_even_revenue = inputs["fixed_costs"] / contribution_margin if contribution_margin > 0 else None
    discounted_roi = (pv - inputs["initial_investment"]) / inputs["initial_investment"] * 100 if inputs["initial_investment"] > 0 else None
    return {
        "npv": npv,
        "irr": irr,
        "irr_error": irr_error,
        "payback_period": payback_period,
        "profitability_index": profitability_index,
        "break_even_revenue": break_even_revenue,
        "discounted_roi": discounted_roi,
        "discounted_flows": discounted_flows,
        "pv": pv
    }

@st.cache_data
def monte_carlo_simulation(inputs, n_simulations=1000):
    npvs = []
    for _ in range(n_simulations):
        sim_inputs = inputs.copy()
        sim_inputs["revenue_growth"] *= np.random.normal(1, 0.1)
        sim_inputs["variable_cost_percent"] *= np.random.normal(1, 0.05)
        _, revenues, cash_flows, _ = calculate_cash_flows(sim_inputs)
        _, after_tax_flows = calculate_taxes(sim_inputs, revenues, cash_flows)
        metrics = calculate_financial_metrics(sim_inputs, after_tax_flows)
        npvs.append(metrics["npv"])
    return np.percentile(npvs, [5, 50, 95]), np.array(npvs)

# ------------- UI Sections -------------

def display_financial_summary(inputs, metrics, currency_symbol):
    st.subheader("📊 Financial Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div class="metric-card high-contrast" role="region" aria-label="Net Present Value">
                <h4>Net Present Value (NPV)</h4>
                <h3 style="color: {'#e74c3c' if metrics['npv'] < 0 else '#27ae60'}">{fmt_curr(metrics['npv'], currency_symbol)}</h3>
                <p>Discount Rate: {fmt_pct(inputs['discount_rate'])}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card high-contrast" role="region" aria-label="Internal Rate of Return">
                <h4>Internal Rate of Return (IRR)</h4>
                <h3>{fmt_pct(metrics['irr']) if metrics['irr'] is not None else (metrics['irr_error'])}</h3>
                <p>Target: {fmt_pct(inputs['discount_rate'])}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card high-contrast" role="region" aria-label="Payback Period">
                <h4>Payback Period</h4>
                <h3>{fmt_yrs(metrics['payback_period'])}</h3>
                <p>Project Lifespan: {inputs['project_lifespan']} years</p>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="metric-card high-contrast" role="region" aria-label="Profitability Index">
                <h4>Profitability Index</h4>
                <h3>{metrics['profitability_index'] and f"{metrics['profitability_index']:.2f}" or 'N/A'}</h3>
                <p>Present Value: {fmt_curr(metrics['pv'], currency_symbol)}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card high-contrast" role="region" aria-label="Discounted ROI">
                <h4>Discounted ROI</h4>
                <h3>{fmt_pct(metrics['discounted_roi'])}</h3>
                <p>Initial Investment: {fmt_curr(inputs['initial_investment'], currency_symbol)}</p>
            </div>
            """, unsafe_allow_html=True)
        st.markdown(f"""
            <div class="metric-card high-contrast" role="region" aria-label="Break-even Revenue">
                <h4>Break-even Revenue</h4>
                <h3>{fmt_curr(metrics['break_even_revenue'], currency_symbol)}</h3>
                <p>Year 1 Revenue: {fmt_curr(inputs['projected_revenue'], currency_symbol)}</p>
            </div>
            """, unsafe_allow_html=True)
    st.markdown("---")
    if (metrics["npv"] > 0 and 
        (metrics["irr"] is not None and metrics["irr"] > inputs["discount_rate"]) and 
        metrics["payback_period"] and metrics["payback_period"] <= inputs["project_lifespan"]):
        st.success("✅ **Financially Viable**: This expansion appears profitable based on current projections.")
    else:
        reasons = []
        if metrics["npv"] <= 0:
            reasons.append("Negative or zero NPV")
        if metrics["irr"] is not None and metrics["irr"] < inputs["discount_rate"]:
            reasons.append(f"IRR below target rate ({fmt_pct(inputs['discount_rate'])})")
        if not metrics["payback_period"] or metrics["payback_period"] > inputs["project_lifespan"]:
            reasons.append("Payback period exceeds project lifespan or is not achieved")
        st.warning(f"⚠ **Review Recommended**: {'; '.join(reasons)}. Consider adjusting your financial parameters.")

def display_cash_flow_table(years, revenues, cash_flows, tax_rates, after_tax_flows, discounted_flows, currency_symbol):
    st.subheader("📋 Annual Cash Flow Projections")
    data = {
        "Year": years,
        "Revenue": revenues,
        "Pre-tax Cash Flow": cash_flows,
        "Tax Rate (%)": tax_rates,
        "After-tax Cash Flow": after_tax_flows,
        "Discounted Cash Flow": discounted_flows
    }
    df = pd.DataFrame(data)
    display_df = df.copy()
    for col in ["Revenue", "Pre-tax Cash Flow", "After-tax Cash Flow", "Discounted Cash Flow"]:
        display_df[col] = display_df[col].apply(lambda x: fmt_curr(x, currency_symbol))
    display_df["Tax Rate (%)"] = display_df["Tax Rate (%)"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(display_df.style.map(
        lambda x: 'color: #e74c3c' if isinstance(x, str) and '-' in x else '',
        subset=pd.IndexSlice[:, ["Revenue", "Pre-tax Cash Flow", "After-tax Cash Flow", "Discounted Cash Flow"]]),
        height=400, use_container_width=True
    )
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Cash Flow Data (CSV)", data=csv, file_name="uae_expansion_cash_flows.csv", mime="text/csv", use_container_width=True)

def display_charts(years, revenues, cash_flows, after_tax_flows, discounted_flows, inputs, metrics, currency_symbol):
    st.subheader("📈 Financial Visualizations")
    # Cumulative cash flow
    cumulative_flows = np.cumsum(np.concatenate(([-inputs["initial_investment"]], after_tax_flows)))
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=np.concatenate(([0], years)),
        y=cumulative_flows,
        mode='lines+markers',
        name='Cumulative Cash Flow',
        line=dict(width=3, color='#3498db'),
        marker=dict(size=8, color='#2980b9')
    ))
    fig1.add_hline(y=0, line_dash="dash", line_color="#e74c3c", annotation_text="Break-even", annotation_position="bottom right")
    fig1.update_layout(title="Cumulative Cash Flow Over Time", xaxis_title="Year", yaxis_title=f"Amount ({inputs['currency']})",
                       template="plotly_white", hovermode="x unified", title_x=0.5)
    st.plotly_chart(fig1, use_container_width=True)

    fixed_costs_yearly = [inputs["fixed_costs"] * (1 + inputs["inflation_rate"] / 100) ** year for year in range(inputs["project_lifespan"])]
    variable_costs_yearly = [rev * (inputs["variable_cost_percent"] / 100) for rev in revenues]
    costs = [fc + vc for fc, vc in zip(fixed_costs_yearly, variable_costs_yearly)]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=years, y=revenues, name='Revenue', marker_color='#2ecc71'))
    fig2.add_trace(go.Bar(x=years, y=costs, name='Total Costs', marker_color='#e74c3c'))
    fig2.update_layout(barmode='group', title="Annual Revenue vs Operating Costs", xaxis_title="Year", yaxis_title=f"Amount ({inputs['currency']})", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🔍 NPV Sensitivity to Discount Rate")
    discount_rates = np.linspace(max(0.1, inputs["discount_rate"] - 5), inputs["discount_rate"] + 5, 11)
    npvs = [np.sum(after_tax_flows / (1 + r/100) ** np.arange(1, inputs["project_lifespan"] + 1)) - inputs["initial_investment"] for r in discount_rates]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(x=discount_rates, y=npvs, mode='lines+markers', line=dict(width=3, color='#f39c12'), marker=dict(size=8, color='#d35400')))
    fig3.add_hline(y=0, line_dash="dash", line_color="#7f8c8d")
    fig3.add_vline(x=inputs["discount_rate"], line_dash="dash", line_color="#2ecc71", annotation_text="Current Rate")
    fig3.update_layout(title="NPV Sensitivity to Discount Rate Changes", xaxis_title="Discount Rate (%)", yaxis_title=f"NPV ({inputs['currency']})", template="plotly_white")
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("🎲 Monte Carlo Simulation")
    if st.button("Run NPV Uncertainty Analysis", use_container_width=True):
        with st.spinner("Running Monte Carlo simulation..."):
            npv_percentiles, npv_sims = monte_carlo_simulation(inputs)
            st.markdown(f"""
                <div class="metric-card high-contrast" role="region" aria-label="Monte Carlo NPV Results">
                    <h4>NPV Uncertainty (1000 Simulations)</h4>
                    <p>5th Percentile: {fmt_curr(npv_percentiles[0], currency_symbol)}</p>
                    <p>Median: {fmt_curr(npv_percentiles[1], currency_symbol)}</p>
                    <p>95th Percentile: {fmt_curr(npv_percentiles[2], currency_symbol)}</p>
                </div>
                """, unsafe_allow_html=True)
            st.info("This simulation models NPV uncertainty by varying revenue growth (±10%) and variable costs (±5%).")

# ------------- Risk Management (UI + helpers) -------------

def collect_risks_and_ratings():
    """
    Returns a list of dicts:
    [{'risk': str, 'likelihood': int 1-5, 'impact': int 1-5, 'score': int, 'mitigation': str, 'level': 'Low/Medium/High'}]
    """
    st.subheader("⚠️ Risk Management")
    risk_options = list(RISK_MITIGATIONS.keys())
    selected_risks = st.multiselect("Select potential risks to analyze", risk_options, default=["Market Competition", "Talent Acquisition"])

    risk_rows = []
    if selected_risks:
        cols = st.columns([3, 1, 1, 1])
        cols[0].markdown("**Risk**")
        cols[1].markdown("**Likelihood (1-5)**")
        cols[2].markdown("**Impact (1-5)**")
        cols[3].markdown("**Score**")

        for r in selected_risks:
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            like = c2.slider(f"Likelihood - {r}", min_value=1, max_value=5, value=3, key=f"like_{r}")
            imp = c3.slider(f"Impact - {r}", min_value=1, max_value=5, value=3, key=f"imp_{r}")
            score = like * imp
            c1.write(r)
            c4.write(f"{score}")
            level = "High" if score >= 16 else ("Medium" if score >= 9 else "Low")
            risk_rows.append({
                "risk": r,
                "likelihood": like,
                "impact": imp,
                "score": score,
                "level": level,
                "mitigation": RISK_MITIGATIONS[r]
            })

        # Ranked table
        df_risk = pd.DataFrame(risk_rows).sort_values(by=["score", "impact", "likelihood"], ascending=False)
        st.markdown("**Ranked Risks**")
        st.dataframe(df_risk[["risk", "likelihood", "impact", "score", "level"]], use_container_width=True)
        return df_risk
    return pd.DataFrame(columns=["risk","likelihood","impact","score","level","mitigation"])

# ------------- Chart generators for PDF (Matplotlib) -------------

def save_cumulative_cf_chart(path, years, after_tax_flows, initial_investment, currency):
    cum = np.cumsum(np.concatenate(([-initial_investment], after_tax_flows)))
    x = np.concatenate(([0], years))
    plt.figure()
    plt.plot(x, cum, marker="o")
    plt.axhline(0, linestyle="--")
    plt.title("Cumulative Cash Flow Over Time")
    plt.xlabel("Year"); plt.ylabel(f"Amount ({currency})")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_revenue_costs_chart(path, years, revenues, fixed_costs_yearly, variable_cost_percent, currency):
    variable_costs_yearly = [rev * (variable_cost_percent / 100) for rev in revenues]
    total_costs = [fc + vc for fc, vc in zip(fixed_costs_yearly, variable_costs_yearly)]
    plt.figure()
    idx = np.arange(len(years))
    width = 0.35
    plt.bar(idx - width/2, revenues, width, label="Revenue")
    plt.bar(idx + width/2, total_costs, width, label="Total Costs")
    plt.xticks(idx, years)
    plt.title("Annual Revenue vs Operating Costs")
    plt.xlabel("Year"); plt.ylabel(f"Amount ({currency})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_npv_sensitivity_chart(path, after_tax_flows, discount_rate, lifespan, initial_investment, currency):
    drs = np.linspace(max(0.1, discount_rate - 5), discount_rate + 5, 11)
    npvs = [np.sum(after_tax_flows / (1 + r/100) ** np.arange(1, lifespan + 1)) - initial_investment for r in drs]
    plt.figure()
    plt.plot(drs, npvs, marker="o")
    plt.axhline(0, linestyle="--")
    plt.axvline(discount_rate, linestyle="--")
    plt.title("NPV Sensitivity to Discount Rate Changes")
    plt.xlabel("Discount Rate (%)"); plt.ylabel(f"NPV ({currency})")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_monte_carlo_hist(path, npv_sims, currency):
    plt.figure()
    plt.hist(npv_sims, bins=30)
    plt.title("Monte Carlo Simulation: NPV Distribution")
    plt.xlabel(f"NPV ({currency})"); plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

def save_risk_matrix_chart(path, df_risk):
    # heatmap-style 5x5 grid + points
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax.set_xlim(0.5, 5.5); ax.set_ylim(0.5, 5.5)
    ax.set_xlabel("Likelihood"); ax.set_ylabel("Impact")
    ax.set_xticks([1,2,3,4,5]); ax.set_yticks([1,2,3,4,5])
    # background squares
    for x in range(1,6):
        for y in range(1,6):
            score = x*y
            if score >= 16:
                color = (1.0, 0.8, 0.8)  # light red
            elif score >= 9:
                color = (1.0, 0.94, 0.8) # light orange
            else:
                color = (0.88, 1.0, 0.88) # light green
            ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color=color, ec='white'))
    # plot risks
    for _, row in df_risk.iterrows():
        ax.scatter(row["likelihood"], row["impact"], s=80, edgecolors='black')
        ax.text(row["likelihood"]+0.08, row["impact"]+0.08, row["risk"], fontsize=8)
    ax.set_title("Risk Matrix (Likelihood vs Impact)")
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()

# ------------- Report Generation -------------

def generate_report(inputs, metrics, years, revenues, cash_flows, tax_rates, after_tax_flows, fixed_costs_yearly):
    """Generates a professional PDF report with tables + charts (financial + risk matrix)."""
    st.subheader("📄 Generate Professional Report")

    # --- Risk options & notes in UI ---
    df_risk = collect_risks_and_ratings()
    notes = st.text_area(
        "Additional notes for the report:",
        "The analysis suggests this expansion could be viable with proper execution. "
        "Key risks include market competition and talent acquisition.",
        help="Add custom notes to include in the PDF report. Note: Avoid Arabic characters due to font limitations."
    )

    if st.button("🖨️ Generate Full Expansion Report", use_container_width=True):
        with st.spinner('Generating professional report...'):
            # Prepare chart image files in temp dir
            tmpdir = tempfile.mkdtemp()
            currency_symbol = get_currency_symbol_pdf(inputs['currency'])
            # Financial charts
            path_cum = os.path.join(tmpdir, "cum_cf.png")
            path_rev_cost = os.path.join(tmpdir, "rev_costs.png")
            path_npv = os.path.join(tmpdir, "npv_sensitivity.png")
            path_mc = os.path.join(tmpdir, "monte_carlo.png")
            # Risk matrix chart if risks available
            path_risk = os.path.join(tmpdir, "risk_matrix.png") if not df_risk.empty else None

            save_cumulative_cf_chart(path_cum, years, after_tax_flows, inputs["initial_investment"], inputs["currency"])
            save_revenue_costs_chart(path_rev_cost, years, revenues, fixed_costs_yearly, inputs["variable_cost_percent"], inputs["currency"])
            save_npv_sensitivity_chart(path_npv, after_tax_flows, inputs["discount_rate"], inputs["project_lifespan"], inputs["initial_investment"], inputs["currency"])

            # Always run a MC sim for the PDF so it's complete (independent from UI button)
            npv_percentiles, npv_sims = monte_carlo_simulation(inputs)
            save_monte_carlo_hist(path_mc, npv_sims, inputs["currency"])

            if path_risk:
                save_risk_matrix_chart(path_risk, df_risk)

            # Build PDF
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)

            # Title
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="UAE Business Expansion Analysis Report", ln=1, align='C')
            pdf.set_font("Arial", '', 10)
            pdf.cell(200, 8, txt=f"Generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", ln=1, align='C')
            pdf.ln(6)

            # 1. Executive Summary
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="1. Executive Summary", ln=1)
            pdf.set_font("Arial", '', 12)
            irr_text = f"{metrics['irr']:.1f}" if metrics['irr'] is not None else "N/A"
            summary_text = f"""This report analyzes the financial viability of expanding business operations in the UAE.
The analysis covers a {inputs['project_lifespan']}-year period with an initial investment of {fmt_curr(inputs['initial_investment'], currency_symbol)}.
Key financial metrics include a Net Present Value (NPV) of {fmt_curr(metrics['npv'], currency_symbol)} and an Internal Rate of Return (IRR) of {irr_text}% (target: {inputs['discount_rate']:.1f}%)."""
            pdf.multi_cell(0, 8, txt=summary_text)
            pdf.ln(4)

            # 2. Key Financial Metrics (table)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="2. Key Financial Metrics", ln=1)
            pdf.set_font("Arial", '', 12)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(100, 8, txt="Metric", ln=0, border=1, fill=True)
            pdf.cell(90, 8, txt="Value", ln=1, border=1, fill=True)
            metric_data = [
                ("Initial Investment", fmt_curr(inputs['initial_investment'], currency_symbol)),
                ("Project Lifespan", f"{inputs['project_lifespan']} years"),
                ("Discount Rate", f"{inputs['discount_rate']:.1f}%"),
                ("Net Present Value (NPV)", fmt_curr(metrics['npv'], currency_symbol)),
                ("Internal Rate of Return (IRR)", f"{metrics['irr']:.1f}%" if metrics['irr'] is not None else (metrics['irr_error'])),
                ("Payback Period", f"{metrics['payback_period']:.1f} years" if metrics['payback_period'] else "N/A"),
                ("Profitability Index", f"{metrics['profitability_index']:.2f}" if metrics['profitability_index'] else "N/A"),
                ("Break-even Revenue", fmt_curr(metrics['break_even_revenue'], currency_symbol) if metrics['break_even_revenue'] else "N/A")
            ]
            for label, value in metric_data:
                pdf.cell(100, 8, txt=label, ln=0, border=1)
                pdf.cell(90, 8, txt=str(value), ln=1, border=1)
            pdf.ln(4)

            # 3. Annual Cash Flow Projections (table)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="3. Annual Cash Flow Projections", ln=1)
            pdf.set_font("Arial", '', 10)
            pdf.set_fill_color(240, 240, 240)
            pdf.cell(20, 8, txt="Year", ln=0, border=1, fill=True)
            pdf.cell(40, 8, txt="Revenue", ln=0, border=1, fill=True)
            pdf.cell(40, 8, txt="After-tax CF", ln=0, border=1, fill=True)
            pdf.cell(30, 8, txt="Tax Rate", ln=1, border=1, fill=True)
            for i, year in enumerate(years):
                pdf.cell(20, 8, txt=str(year), ln=0, border=1)
                pdf.cell(40, 8, txt=fmt_curr(revenues[i], currency_symbol), ln=0, border=1)
                pdf.cell(40, 8, txt=fmt_curr(after_tax_flows[i], currency_symbol), ln=0, border=1)
                pdf.cell(30, 8, txt=f"{tax_rates[i]:.1f}%", ln=1, border=1)
            pdf.ln(4)

            # 4. Financial Charts (images)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="4. Financial Charts", ln=1)
            pdf.image(path_cum, w=180); pdf.ln(4)
            pdf.image(path_rev_cost, w=180); pdf.ln(4)
            pdf.image(path_npv, w=180); pdf.ln(4)
            pdf.image(path_mc, w=180); pdf.ln(6)

            # 5. Risk Analysis (table + risk matrix)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="5. Risk Analysis", ln=1)
            pdf.set_font("Arial", '', 12)
            if df_risk.empty:
                pdf.multi_cell(0, 8, txt="No risks were selected.")
            else:
                # Risk register table (ranked)
                pdf.set_font("Arial", '', 10)
                pdf.set_fill_color(240, 240, 240)
                pdf.cell(70, 8, txt="Risk", ln=0, border=1, fill=True)
                pdf.cell(20, 8, txt="Like", ln=0, border=1, fill=True)
                pdf.cell(20, 8, txt="Impact", ln=0, border=1, fill=True)
                pdf.cell(20, 8, txt="Score", ln=0, border=1, fill=True)
                pdf.cell(30, 8, txt="Level", ln=1, border=1, fill=True)
                for _, row in df_risk.iterrows():
                    pdf.cell(70, 8, txt=str(row["risk"])[:38], ln=0, border=1)
                    pdf.cell(20, 8, txt=str(int(row["likelihood"])), ln=0, border=1)
                    pdf.cell(20, 8, txt=str(int(row["impact"])), ln=0, border=1)
                    pdf.cell(20, 8, txt=str(int(row["score"])), ln=0, border=1)
                    pdf.cell(30, 8, txt=str(row["level"]), ln=1, border=1)
                pdf.ln(3)
                # Mitigations (top 5)
                pdf.set_font("Arial", 'B', 12)
                pdf.cell(200, 6, txt="Mitigation Strategies (Top 5 by Score)", ln=1)
                pdf.set_font("Arial", '', 11)
                for _, row in df_risk.head(5).iterrows():
                    pdf.multi_cell(0, 6, txt=f"- {row['risk']}: {row['mitigation']}")
                pdf.ln(3)
                # Risk Matrix chart
                if path_risk:
                    pdf.image(path_risk, w=180)
            pdf.ln(6)

            # 6. Conclusion & Recommendations
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="6. Conclusion & Recommendations", ln=1)
            pdf.set_font("Arial", '', 12)
            if (metrics["npv"] > 0 and (metrics["irr"] is not None and metrics["irr"] > inputs["discount_rate"])
                and metrics["payback_period"] and metrics["payback_period"] <= inputs["project_lifespan"]):
                conclusion_text = """Based on the financial analysis, this expansion appears viable. We recommend:
1. Proceeding with detailed market research
2. Developing a phased implementation plan
3. Establishing key performance indicators to monitor progress"""
            else:
                conclusion_text = """The financial analysis suggests caution with this expansion. We recommend:
1. Re-evaluating the financial projections
2. Considering alternative business models
3. Conducting additional market research before committing"""
            pdf.multi_cell(0, 8, txt=conclusion_text)
            pdf.ln(4)

            # 7. Additional Notes
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 8, txt="7. Additional Notes", ln=1)
            pdf.set_font("Arial", '', 12)
            pdf.multi_cell(0, 8, txt=notes)
            pdf.ln(8)

            pdf.set_font("Arial", 'I', 10)
            pdf.cell(0, 8, txt="Generated by UAE Business Expansion Simulator - Confidential", ln=1, align='C')

            # Output
            try:
                pdf_bytes = pdf.output(dest="S").encode('latin-1')
                b64 = base64.b64encode(pdf_bytes).decode()
                href = f'<a href="data:application/octet-stream;base64,{b64}" download="UAE_Expansion_Report.pdf" style="text-decoration: none;"><button style="background-color: #2c3e50; color: white; padding: 10px 20px; border: none; border-radius: 6px; cursor: pointer; font-size: 16px; margin-top: 10px;">Download Full Report</button></a>'
                st.markdown(href, unsafe_allow_html=True)
                st.success("Professional report generated successfully!")
            except Exception as pdf_error:
                st.error(f"Error generating PDF download: {str(pdf_error)}")

# ------------- Main -------------

def main():
    inputs = get_user_inputs()
    is_valid, messages = validate_inputs(inputs)
    if not is_valid:
        for error in messages:
            st.error(error)
        st.warning("Please correct the input errors to proceed with the analysis.")
        return
    else:
        for warning in messages:
            st.warning(warning)

    years, revenues, cash_flows, fixed_costs_yearly = calculate_cash_flows(inputs)
    tax_rates, after_tax_flows = calculate_taxes(inputs, revenues, cash_flows)
    metrics = calculate_financial_metrics(inputs, after_tax_flows)
    currency_symbol = get_currency_symbol(inputs['currency'])

    tab1, tab2, tab3, tab4 = st.tabs(["📊 Summary", "📋 Cash Flows", "📈 Charts", "📄 Report"])

    with tab1:
        display_financial_summary(inputs, metrics, currency_symbol)

    with tab2:
        display_cash_flow_table(
            years, revenues, cash_flows, tax_rates, after_tax_flows, metrics["discounted_flows"], currency_symbol
        )

    with tab3:
        display_charts(
            years, revenues, cash_flows, after_tax_flows, metrics["discounted_flows"], inputs, metrics, currency_symbol
        )

    with tab4:
        generate_report(
            inputs, metrics, years, revenues, cash_flows, tax_rates, after_tax_flows, fixed_costs_yearly
        )

if __name__ == "__main__":
    main()
