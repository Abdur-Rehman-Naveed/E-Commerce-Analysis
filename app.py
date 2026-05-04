import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import norm, poisson
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import textwrap
import warnings
warnings.filterwarnings('ignore')

# --- Page Config ---
st.set_page_config(page_title="Retail Data Intelligence", page_icon="📈", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stMetric {background-color: white; padding: 15px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);}
    h1 {color: #2c3e50; font-weight: bold;}
    h2 {color: #34495e;}
    h3 {color: #1abc9c;}
    .block-container {padding-top: 2rem;}
    </style>
""", unsafe_allow_html=True)

DATA_PATH = 'online_retail_II.xlsx'

@st.cache_data(show_spinner="Loading and Preprocessing 45MB Dataset (This takes ~30s on first run)...")
def load_data():
    sheets = []
    for sheet in ['Year 2009-2010', 'Year 2010-2011']:
        s = pd.read_excel(DATA_PATH, sheet_name=sheet, parse_dates=['InvoiceDate'])
        s['Sheet'] = sheet
        sheets.append(s)
    raw = pd.concat(sheets, ignore_index=True)
    raw.rename(columns={'Invoice': 'InvoiceNo', 'Price': 'UnitPrice', 'Customer ID': 'CustomerID'}, inplace=True)
    
    df = raw.copy()
    df.drop_duplicates(inplace=True)
    df['IsCancellation'] = df['InvoiceNo'].astype(str).str.startswith('C')
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(int)
    df_clean = df[~df['IsCancellation']].copy()
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
    
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.day_name()
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    df_clean = df_clean.reset_index(drop=True)

    ref_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
    rfm = df_clean.groupby('CustomerID').agg(
        Recency=('InvoiceDate', lambda x: (ref_date - x.max()).days),
        Frequency=('InvoiceNo', 'nunique'),
        Monetary=('TotalPrice', 'sum')
    ).reset_index()

    rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1]).astype(int)
    rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
    rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1,2,3,4,5]).astype(int)
    rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

    def segment(score):
        if score >= 13: return 'VIP / Champions'
        elif score >= 10: return 'Loyal Customers'
        elif score >= 7: return 'Potential Loyalists'
        elif score >= 4: return 'At-Risk Customers'
        else: return 'Lost / Churned'

    rfm['Segment'] = rfm['RFM_Score'].apply(segment)
    return df_clean, rfm

# ─────────────────────────────────────────────────────────────────────────────
# MODULES
# ─────────────────────────────────────────────────────────────────────────────

def fmt_currency(val): return f"£{val:,.2f}"
def fmt_num(val, dec=2): return f"{val:,.{dec}f}"

def module_executive_summary(df, rfm):
    st.title("📈 Executive Summary & Key Insights")
    st.markdown("A high-level overview of business performance from Dec 2009 to Dec 2011.")
    
    total_rev = df['TotalPrice'].sum()
    total_orders = df['InvoiceNo'].nunique()
    total_cust = df['CustomerID'].nunique()
    avg_order = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
    repeat_rate = (rfm['Frequency'] > 1).mean() * 100
    vip_rev = rfm[rfm['Segment'] == 'VIP / Champions']['Monetary'].sum()
    vip_share = vip_rev / rfm['Monetary'].sum() * 100

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", fmt_currency(total_rev))
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Unique Customers", f"{total_cust:,}")
    
    col4, col5, col6 = st.columns(3)
    col4.metric("Average Order Value", fmt_currency(avg_order))
    col5.metric("Repeat Purchase Rate", f"{repeat_rate:.1f}%")
    col6.metric("VIP Revenue Share", f"{vip_share:.1f}%")

    st.markdown("### 💡 Strategic Recommendations")
    st.info(f"**1. Retain VIP Customers:** {vip_share:.1f}% of revenue comes from Champions. Launch loyalty rewards.")
    st.warning("**2. Reactivate At-Risk Segment:** Target customers with high recency and low frequency with win-back campaigns.")
    st.success("**3. Capitalise on Seasonal Peaks:** Q4 shows the highest seasonality index. Stock up proactively.")
    
def module_descriptive_stats(df, rfm):
    st.title("📊 Descriptive Statistics")
    
    targets = {
        'Quantity (units)': df['Quantity'],
        'Unit Price (£)': df['UnitPrice'],
        'Total Line Value (£)': df['TotalPrice'],
        'Recency (days)': rfm['Recency'],
        'Frequency': rfm['Frequency'],
        'Monetary (£)': rfm['Monetary'],
    }
    
    stats_data = []
    for label, s in targets.items():
        stats_data.append({
            'Variable': label, 'Mean': s.mean(), 'Median': s.median(), 
            'Std Dev': s.std(), 'Min': s.min(), 'Max': s.max()
        })
    st.dataframe(pd.DataFrame(stats_data).style.format({'Mean': '{:.2f}', 'Median': '{:.2f}', 'Std Dev': '{:.2f}', 'Min': '{:.2f}', 'Max': '{:.2f}'}), use_container_width=True)

    st.subheader("Core Distributions")
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    panels = [
        (df['Quantity'].clip(upper=df['Quantity'].quantile(0.99)), 'Quantity per Line', 'steelblue'),
        (df['UnitPrice'].clip(upper=df['UnitPrice'].quantile(0.99)), 'Unit Price (£)', 'darkorange'),
        (df['TotalPrice'].clip(upper=df['TotalPrice'].quantile(0.99)), 'Total Line Value (£)', 'seagreen'),
        (rfm['Recency'], 'Recency (days)', 'mediumpurple'),
        (rfm['Frequency'].clip(upper=rfm['Frequency'].quantile(0.99)), 'Purchase Frequency', 'crimson'),
        (rfm['Monetary'].clip(upper=rfm['Monetary'].quantile(0.99)), 'Monetary Value (£)', 'goldenrod'),
    ]
    for ax, (series, title, color) in zip(axes.flat, panels):
        ax.hist(series, bins=50, color=color, alpha=0.85, edgecolor='white')
        ax.axvline(series.mean(), color='black', lw=1.5, linestyle='--', label=f"Mean={series.mean():.1f}")
        ax.axvline(series.median(), color='red', lw=1.5, linestyle=':', label=f"Median={series.median():.1f}")
        ax.set_title(title)
        ax.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

def module_probability(df, rfm):
    st.title("🎲 Probability & Distributions")
    
    total_customers = rfm['CustomerID'].nunique()
    repeat = (rfm['Frequency'] > 1).sum()
    p_repeat = repeat / total_customers
    
    col1, col2 = st.columns(2)
    col1.metric("P(Repeat Purchase)", f"{p_repeat*100:.2f}%")
    col2.metric("P(First-time Only)", f"{(1-p_repeat)*100:.2f}%")
    
    st.subheader("Purchase Frequency (Poisson Distribution)")
    lam = rfm['Frequency'].mean()
    max_k = int(rfm['Frequency'].quantile(0.95))
    k_vals = np.arange(0, max_k + 1)
    poisson_probs = poisson.pmf(k_vals, lam)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(k_vals[:15], poisson_probs[:15], color='steelblue', alpha=0.8, edgecolor='white')
    ax.set_title(f'Poisson PMF (λ={lam:.2f})')
    ax.set_xlabel('Number of Orders (k)')
    ax.set_ylabel('Probability P(X=k)')
    st.pyplot(fig)

    st.subheader("Order Value vs. Normal Distribution")
    order_val = df.groupby('InvoiceNo')['TotalPrice'].sum()
    ov_clipped = order_val.clip(upper=order_val.quantile(0.99))
    mu, sigma = ov_clipped.mean(), ov_clipped.std()
    
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    x_range = np.linspace(ov_clipped.min(), ov_clipped.max(), 300)
    ax2.hist(ov_clipped, bins=60, density=True, color='darkorange', alpha=0.7, label='Observed')
    ax2.plot(x_range, norm.pdf(x_range, mu, sigma), 'k-', lw=2, label='Normal Fit')
    ax2.set_title(f'Normal Fit (μ=£{mu:.2f}, σ=£{sigma:.2f})')
    ax2.legend()
    st.pyplot(fig2)

def module_confidence(df, rfm):
    st.title("🛡️ Confidence Intervals")
    
    def get_ci(data, alpha=0.05):
        n, mean, se = len(data), data.mean(), stats.sem(data)
        margin = stats.t.ppf(1 - alpha/2, n-1) * se
        return mean, mean - margin, mean + margin, se, n

    targets = {
        'Avg Order Value (£)': df.groupby('InvoiceNo')['TotalPrice'].sum(),
        'Avg Items per Order': df.groupby('InvoiceNo')['Quantity'].sum(),
        'Avg Customer Spend (£)': rfm['Monetary'],
        'Avg Purchase Frequency': rfm['Frequency'],
    }
    
    rows = []
    for label, series in targets.items():
        m, lo, hi, se, n = get_ci(series, 0.05)
        rows.append({'Metric': label, 'Point Estimate': m, '95% Lower': lo, '95% Upper': hi})
        
    st.dataframe(pd.DataFrame(rows).style.format({'Point Estimate': '{:.2f}', '95% Lower': '{:.2f}', '95% Upper': '{:.2f}'}), use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    means = [r['Point Estimate'] for r in rows]
    err_lo = [r['Point Estimate'] - r['95% Lower'] for r in rows]
    err_hi = [r['95% Upper'] - r['Point Estimate'] for r in rows]
    labels = [r['Metric'] for r in rows]
    
    ax.barh(labels, means, xerr=[err_lo, err_hi], color='steelblue', alpha=0.75, capsize=5)
    ax.set_title('95% Confidence Intervals')
    st.pyplot(fig)

def module_regression(df, rfm):
    st.title("📈 Regression & Prediction")
    
    rfm_feat = rfm.copy()
    rfm_feat['Log_Monetary'] = np.log1p(rfm_feat['Monetary'])
    rfm_feat['Log_Frequency'] = np.log1p(rfm_feat['Frequency'])

    X = rfm_feat[['Recency','Log_Frequency','R_Score','F_Score','M_Score']].values
    y = rfm_feat['Log_Monetary'].values
    
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    ols = LinearRegression()
    ols.fit(X_tr_s, y_tr)
    y_pred = ols.predict(X_te_s)
    
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    r2 = r2_score(y_te, y_pred)
    
    col1, col2 = st.columns(2)
    col1.metric("Model RMSE (Log-Spend)", f"{rmse:.4f}")
    col2.metric("Model R² (Accuracy)", f"{r2:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    residuals = y_te - y_pred
    axes[0].scatter(y_pred, residuals, alpha=0.3, color='steelblue', s=15)
    axes[0].axhline(0, color='red', lw=1.5, linestyle='--')
    axes[0].set_title('Residuals vs. Fitted')
    
    axes[1].scatter(y_te, y_pred, alpha=0.3, color='darkorange', s=15)
    lims = [min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())]
    axes[1].plot(lims, lims, 'k--', lw=1.5)
    axes[1].set_title('Actual vs. Predicted')
    
    st.pyplot(fig)

def module_segmentation(df, rfm):
    st.title("🎯 Customer Segmentation (K-Means)")
    
    X = np.log1p(rfm[['Recency','Frequency','Monetary']].values)
    X_s = StandardScaler().fit_transform(X)
    
    km = KMeans(n_clusters=5, random_state=42, n_init=15)
    rfm['Cluster'] = km.fit_predict(X_s)
    
    cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
    labels_ordered = ['VIP / Champions', 'Loyal Customers', 'Potential Loyalists', 'At-Risk Customers', 'Lost / Churned']
    cluster_map = dict(zip(cluster_means.index, labels_ordered))
    rfm['KMeans_Segment'] = rfm['Cluster'].map(cluster_map)
    
    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    palette = ['#2ecc71','#3498db','#f39c12','#e74c3c','#9b59b6']
    
    ax1 = fig.add_subplot(gs[0, 0])
    for idx, (seg, grp) in enumerate(rfm.groupby('KMeans_Segment')):
        ax1.scatter(np.log1p(grp['Recency']), np.log1p(grp['Monetary']), s=20, alpha=0.5, label=seg, color=palette[idx%5])
    ax1.set_xlabel('log(Recency + 1)')
    ax1.set_ylabel('log(Monetary + 1)')
    ax1.legend(fontsize=8)
    ax1.set_title("Clusters: Recency vs Spend")

    ax2 = fig.add_subplot(gs[0, 1])
    seg_counts = rfm['KMeans_Segment'].value_counts()
    ax2.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%', colors=palette, startangle=90)
    ax2.set_title('Segment Distribution')
    
    st.pyplot(fig)

def module_timeseries(df):
    st.title("📅 Time Series & Seasonality")
    
    monthly = df.groupby('YearMonth').agg(Revenue=('TotalPrice','sum')).reset_index()
    monthly['YearMonth_str'] = monthly['YearMonth'].astype(str)
    
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly['YearMonth_str'], monthly['Revenue'], marker='o', color='steelblue', lw=2)
    ax.fill_between(monthly['YearMonth_str'], monthly['Revenue'], alpha=0.2, color='steelblue')
    plt.xticks(rotation=45)
    ax.set_title("Monthly Revenue Trend")
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day_rev = df.groupby('DayOfWeek')['TotalPrice'].sum().reindex(day_order).dropna()
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.bar(day_order, day_rev.values, color='seagreen', alpha=0.8)
        plt.xticks(rotation=45)
        st.pyplot(fig2)

    with col2:
        st.subheader("Quarterly Seasonality")
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        q_rev = df.groupby('Quarter')['TotalPrice'].sum()
        q_idx = q_rev / q_rev.mean()
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.bar([f"Q{q}" for q in q_idx.index], q_idx.values, color=['#e74c3c' if v<1 else '#2ecc71' for v in q_idx.values])
        ax3.axhline(1.0, color='black', linestyle='--')
        st.pyplot(fig3)

def module_product_geo(df):
    st.title("🌍 Product & Geographic Insights")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 10 Products")
        top_prod = df.groupby(['StockCode','Description']).agg(Rev=('TotalPrice','sum')).sort_values('Rev', ascending=False).head(10).reset_index()
        st.dataframe(top_prod.style.format({'Rev': '£{:.2f}'}))
        
    with col2:
        st.subheader("Top 10 Countries")
        top_geo = df.groupby('Country').agg(Rev=('TotalPrice','sum')).sort_values('Rev', ascending=False).head(10).reset_index()
        st.dataframe(top_geo.style.format({'Rev': '£{:.2f}'}))

# --- Main App ---
def main():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/3144/3144456.png", width=100)
        st.title("Navigation")
        menu = st.radio("Select Module", [
            "Executive Summary",
            "Descriptive Statistics",
            "Probability & Distributions",
            "Confidence Intervals",
            "Regression & Prediction",
            "Customer Segmentation",
            "Time Series & Seasonality",
            "Product & Geo Insights"
        ])
        st.markdown("---")
        st.markdown("Created for **Retail Analysis**")

    try:
        df, rfm = load_data()
    except Exception as e:
        st.error(f"Error loading data: {e}. Please ensure 'online_retail_II.xlsx' is in the directory.")
        return

    if menu == "Executive Summary": module_executive_summary(df, rfm)
    elif menu == "Descriptive Statistics": module_descriptive_stats(df, rfm)
    elif menu == "Probability & Distributions": module_probability(df, rfm)
    elif menu == "Confidence Intervals": module_confidence(df, rfm)
    elif menu == "Regression & Prediction": module_regression(df, rfm)
    elif menu == "Customer Segmentation": module_segmentation(df, rfm)
    elif menu == "Time Series & Seasonality": module_timeseries(df)
    elif menu == "Product & Geo Insights": module_product_geo(df)

if __name__ == '__main__':
    main()
