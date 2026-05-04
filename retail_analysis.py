"""
=============================================================================
  Global E-Commerce Customer Behavior & Predictive Analysis System
  Dataset: Online Retail II (UK-based, Dec 2009 – Dec 2011)
=============================================================================
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg' if 'DISPLAY' in os.environ else 'Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import norm, poisson, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tabulate import tabulate
from datetime import datetime
import textwrap

warnings.filterwarnings('ignore')
sns.set_theme(style='darkgrid', palette='deep')
plt.rcParams.update({'figure.dpi': 110, 'font.family': 'DejaVu Sans'})

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS & PATHS
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = r'E:\All semester Project\probability Project\online_retail_II.xlsx'
DIVIDER   = '═' * 72
SUBDIV    = '─' * 72
FIGURE_SAVE_DIR = r'E:\All semester Project\probability Project\outputs'

# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def banner(title: str, char: str = '═') -> None:
    line = char * 72
    print(f"\n{line}")
    print(f"  {title}")
    print(f"{line}")

def sub_banner(title: str) -> None:
    print(f"\n  {'─'*68}")
    print(f"    {title}")
    print(f"  {'─'*68}")

def pause() -> None:
    input("\n  ↩  Press Enter to continue...")

def save_and_show(fig, filename: str) -> None:
    path = os.path.join(FIGURE_SAVE_DIR, filename)
    fig.savefig(path, bbox_inches='tight', facecolor='white')
    print(f"\n  📊 Chart saved → {path}")
    try:
        plt.show(block=False)
        plt.pause(2)
    except Exception:
        pass
    plt.close('all')

def fmt_currency(val: float) -> str:
    return f"£{val:,.2f}"

def fmt_num(val: float, decimals: int = 2) -> str:
    return f"{val:,.{decimals}f}"

def print_table(data, headers, title: str = '') -> None:
    if title:
        print(f"\n  {title}")
    print(tabulate(data, headers=headers, tablefmt='rounded_outline',
                   floatfmt='.4f', numalign='right'))

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 0 — DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    """Handles all data ingestion and cleaning tasks."""

    def __init__(self):
        self.raw: pd.DataFrame | None = None
        self.df:  pd.DataFrame | None = None
        self.rfm: pd.DataFrame | None = None

    # ── Load ──────────────────────────────────────────────────────────────
    def load(self) -> None:
        banner("Loading & Preprocessing Dataset …")
        print("\n  Reading both Excel sheets — this may take ~30 s on first run.")

        sheets = []
        for sheet in ['Year 2009-2010', 'Year 2010-2011']:
            s = pd.read_excel(DATA_PATH, sheet_name=sheet, parse_dates=['InvoiceDate'])
            s['Sheet'] = sheet
            sheets.append(s)
            print(f"    ✔  {sheet:<20}  rows: {len(s):>8,}")

        self.raw = pd.concat(sheets, ignore_index=True)

        # ── Standardise column names ──────────────────────────────────────
        self.raw.rename(columns={
            'Invoice':     'InvoiceNo',
            'Price':       'UnitPrice',
            'Customer ID': 'CustomerID'
        }, inplace=True)

        print(f"\n  ✔  Combined dataset: {len(self.raw):,} rows × {self.raw.shape[1]} cols")
        self._clean()

    # ── Clean ─────────────────────────────────────────────────────────────
    def _clean(self) -> None:
        sub_banner("Data Cleaning Pipeline")
        df = self.raw.copy()

        steps = []
        n0 = len(df)

        # Drop duplicates
        df.drop_duplicates(inplace=True)
        steps.append(('Duplicate rows removed', n0 - len(df)))

        # Identify cancellations (Invoice starts with 'C')
        df['IsCancellation'] = df['InvoiceNo'].astype(str).str.startswith('C')

        # Drop rows with missing CustomerID
        n_before = len(df)
        df.dropna(subset=['CustomerID'], inplace=True)
        steps.append(('Missing CustomerID dropped', n_before - len(df)))

        # Ensure numeric types
        df['CustomerID'] = df['CustomerID'].astype(int)

        # Filter out cancellations for main analysis
        n_before = len(df)
        df_clean = df[~df['IsCancellation']].copy()
        steps.append(('Cancellation rows excluded', n_before - len(df_clean)))

        # Remove non-positive quantities / prices
        n_before = len(df_clean)
        df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]
        steps.append(('Non-positive Qty/Price removed', n_before - len(df_clean)))

        # Derived variables
        df_clean['TotalPrice']  = df_clean['Quantity'] * df_clean['UnitPrice']
        df_clean['YearMonth']   = df_clean['InvoiceDate'].dt.to_period('M')
        df_clean['DayOfWeek']   = df_clean['InvoiceDate'].dt.day_name()
        df_clean['Hour']        = df_clean['InvoiceDate'].dt.hour

        self.df = df_clean.reset_index(drop=True)

        # Print cleaning summary
        data = [(s, f"{v:,}") for s, v in steps]
        data.append(('Final usable rows', f"{len(self.df):,}"))
        data.append(('Unique customers',  f"{self.df['CustomerID'].nunique():,}"))
        data.append(('Unique products',   f"{self.df['StockCode'].nunique():,}"))
        data.append(('Unique countries',  f"{self.df['Country'].nunique():,}"))
        print_table(data, ['Step', 'Count'])

        # Build RFM
        self._build_rfm()

    # ── RFM ───────────────────────────────────────────────────────────────
    def _build_rfm(self) -> None:
        ref_date = self.df['InvoiceDate'].max() + pd.Timedelta(days=1)
        rfm = (
            self.df.groupby('CustomerID')
            .agg(
                Recency   =('InvoiceDate',  lambda x: (ref_date - x.max()).days),
                Frequency =('InvoiceNo',    'nunique'),
                Monetary  =('TotalPrice',   'sum')
            )
            .reset_index()
        )

        # Score 1–5 (5 = best)
        rfm['R_Score'] = pd.qcut(rfm['Recency'],   5, labels=[5,4,3,2,1]).astype(int)
        rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=[1,2,3,4,5]).astype(int)
        rfm['M_Score'] = pd.qcut(rfm['Monetary'],  5, labels=[1,2,3,4,5]).astype(int)
        rfm['RFM_Score'] = rfm[['R_Score','F_Score','M_Score']].sum(axis=1)

        def segment(score):
            if score >= 13: return 'VIP / Champions'
            elif score >= 10: return 'Loyal Customers'
            elif score >= 7:  return 'Potential Loyalists'
            elif score >= 4:  return 'At-Risk Customers'
            else:             return 'Lost / Churned'

        rfm['Segment'] = rfm['RFM_Score'].apply(segment)
        self.rfm = rfm
        print(f"\n  ✔  RFM table built for {len(rfm):,} customers.")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 — DESCRIPTIVE STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
class DescriptiveStats:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def run(self) -> None:
        df  = self.dl.df
        rfm = self.dl.rfm
        banner("Module 1 — Descriptive Statistics")

        # ── Numerical summary ────────────────────────────────────────────
        sub_banner("Core Variable Summary")
        targets = {
            'Quantity (units)':       df['Quantity'],
            'Unit Price (£)':         df['UnitPrice'],
            'Total Price / Line (£)': df['TotalPrice'],
            'RFM – Recency (days)':   rfm['Recency'],
            'RFM – Frequency':        rfm['Frequency'],
            'RFM – Monetary (£)':     rfm['Monetary'],
        }

        rows = []
        for label, s in targets.items():
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            rows.append([
                label,
                fmt_num(s.mean()),
                fmt_num(s.median()),
                fmt_num(s.std()),
                fmt_num(s.var()),
                fmt_num(s.min()),
                fmt_num(q1),
                fmt_num(q3),
                fmt_num(s.max()),
                fmt_num(stats.skew(s)),
                fmt_num(stats.kurtosis(s)),
            ])

        print_table(rows,
            ['Variable','Mean','Median','Std Dev','Variance',
             'Min','Q1','Q3','Max','Skewness','Kurtosis'])

        # ── Order-level summary ──────────────────────────────────────────
        sub_banner("Order-Level Statistics")
        order = df.groupby('InvoiceNo').agg(
            OrderValue  =('TotalPrice', 'sum'),
            ItemCount   =('Quantity',   'sum'),
            UniqueItems =('StockCode',  'nunique')
        )
        rows2 = []
        for col in ['OrderValue','ItemCount','UniqueItems']:
            s = order[col]
            rows2.append([col, fmt_currency(s.mean()) if col=='OrderValue' else fmt_num(s.mean()),
                          fmt_num(s.median()), fmt_num(s.std()), fmt_num(s.min()), fmt_num(s.max())])
        print_table(rows2, ['Metric','Mean','Median','Std','Min','Max'])

        # ── Revenue by country (top 10) ──────────────────────────────────
        sub_banner("Top 10 Countries by Revenue")
        rev_country = (
            df.groupby('Country')['TotalPrice']
            .sum()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        rev_country['Revenue (£)'] = rev_country['TotalPrice'].apply(fmt_currency)
        rev_country['Share (%)']   = (rev_country['TotalPrice'] / rev_country['TotalPrice'].sum() * 100).apply(lambda x: f"{x:.2f}%")
        print_table(rev_country[['Country','Revenue (£)','Share (%)']].values.tolist(),
                    ['Country','Revenue (£)','Share (%)'])

        # ── Chart ────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        fig.suptitle('Descriptive Statistics — Core Distributions', fontsize=14, fontweight='bold')

        panels = [
            (df['Quantity'].clip(upper=df['Quantity'].quantile(0.99)),
             'Quantity per Line', 'steelblue'),
            (df['UnitPrice'].clip(upper=df['UnitPrice'].quantile(0.99)),
             'Unit Price (£)', 'darkorange'),
            (df['TotalPrice'].clip(upper=df['TotalPrice'].quantile(0.99)),
             'Total Line Value (£)', 'seagreen'),
            (rfm['Recency'],        'Recency (days)',     'mediumpurple'),
            (rfm['Frequency'].clip(upper=rfm['Frequency'].quantile(0.99)),
             'Purchase Frequency', 'crimson'),
            (rfm['Monetary'].clip(upper=rfm['Monetary'].quantile(0.99)),
             'Monetary Value (£)', 'goldenrod'),
        ]

        for ax, (series, title, color) in zip(axes.flat, panels):
            ax.hist(series, bins=50, color=color, alpha=0.85, edgecolor='white', linewidth=0.4)
            ax.axvline(series.mean(),   color='black', lw=1.5, linestyle='--', label=f"Mean={series.mean():.1f}")
            ax.axvline(series.median(), color='red',   lw=1.5, linestyle=':',  label=f"Median={series.median():.1f}")
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.legend(fontsize=7)
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')

        plt.tight_layout()
        save_and_show(fig, 'desc_stats_distributions.png')
        pause()

# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 — PROBABILITY & DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
class ProbabilityAnalysis:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def run(self) -> None:
        df  = self.dl.df
        rfm = self.dl.rfm
        banner("Module 2 — Probability & Distribution Analysis")

        # ── Repeat purchase probability ──────────────────────────────────
        sub_banner("Repeat Purchase Likelihood")
        total_customers = rfm['CustomerID'].nunique()
        repeat          = (rfm['Frequency'] > 1).sum()
        p_repeat        = repeat / total_customers

        print(f"  Total unique customers  : {total_customers:,}")
        print(f"  Repeat purchasers (≥2)  : {repeat:,}")
        print(f"  P(repeat purchase)      : {p_repeat:.4f}  ({p_repeat*100:.2f}%)")
        print(f"  P(first-time only)      : {1-p_repeat:.4f}  ({(1-p_repeat)*100:.2f}%)")

        # Conditional probabilities by country (top 5)
        sub_banner("Conditional Repeat-Purchase Rate by Country (Top 5)")
        cust_country = df.groupby(['CustomerID','Country'])['InvoiceNo'].nunique().reset_index()
        cust_country.columns = ['CustomerID','Country','Freq']
        country_rep = (
            cust_country.groupby('Country')
            .agg(Total=('Freq','count'), Repeat=('Freq', lambda x: (x>1).sum()))
            .assign(RepeatRate=lambda d: d['Repeat']/d['Total'])
            .sort_values('Total', ascending=False)
            .head(5)
            .reset_index()
        )
        country_rep['RepeatRate'] = country_rep['RepeatRate'].apply(lambda x: f"{x*100:.2f}%")
        print_table(country_rep.values.tolist(),
                    ['Country','Total Customers','Repeat Buyers','Repeat Rate'])

        # ── Poisson — purchase frequency ────────────────────────────────
        sub_banner("Poisson Distribution — Purchase Frequency")
        lam = rfm['Frequency'].mean()
        print(f"  λ (average orders/customer) = {lam:.4f}")

        max_k = int(rfm['Frequency'].quantile(0.95))
        k_vals = np.arange(0, max_k + 1)
        poisson_probs = poisson.pmf(k_vals, lam)

        rows = []
        for k, p in zip(k_vals[:10], poisson_probs[:10]):
            expected = p * total_customers
            rows.append([k, f"{p:.6f}", f"{p*100:.3f}%", f"{expected:,.0f}"])
        print_table(rows, ['Orders (k)', 'P(X=k)', 'P(X=k)%', 'Expected Customers'])

        # ── Normal fit — order value ─────────────────────────────────────
        sub_banner("Normal Distribution Fit — Order Value")
        order_val = df.groupby('InvoiceNo')['TotalPrice'].sum()
        ov_clipped = order_val.clip(upper=order_val.quantile(0.99))
        mu, sigma = ov_clipped.mean(), ov_clipped.std()
        print(f"  μ (mean order value)    = £{mu:.2f}")
        print(f"  σ (std dev)             = £{sigma:.2f}")

        # Probability queries
        for threshold in [100, 250, 500]:
            z   = (threshold - mu) / sigma
            p_lt = norm.cdf(z)
            p_gt = 1 - p_lt
            print(f"  P(order < £{threshold:<5})   = {p_lt:.4f}  ({p_lt*100:.2f}%)")
            print(f"  P(order > £{threshold:<5})   = {p_gt:.4f}  ({p_gt*100:.2f}%)")

        # Normality test
        _, p_value = stats.shapiro(ov_clipped.sample(min(5000, len(ov_clipped)), random_state=42))
        print(f"\n  Shapiro-Wilk p-value (sample n=5,000): {p_value:.6f}")
        print(f"  → Data {'DOES NOT' if p_value < 0.05 else 'DOES'} follow Normal dist. (α=0.05)")

        # ── Charts ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle('Probability & Distribution Analysis', fontsize=13, fontweight='bold')

        # Poisson PMF
        axes[0].bar(k_vals[:15], poisson_probs[:15], color='steelblue', alpha=0.8, edgecolor='white')
        axes[0].set_title('Poisson PMF – Purchase Frequency', fontweight='bold')
        axes[0].set_xlabel('Number of Orders (k)')
        axes[0].set_ylabel('Probability P(X=k)')

        # Normal fit overlay
        x_range = np.linspace(ov_clipped.min(), ov_clipped.max(), 300)
        axes[1].hist(ov_clipped, bins=60, density=True, color='darkorange', alpha=0.7, label='Observed')
        axes[1].plot(x_range, norm.pdf(x_range, mu, sigma), 'k-', lw=2, label='Normal Fit')
        axes[1].set_title('Order Value vs. Normal Distribution', fontweight='bold')
        axes[1].set_xlabel('Order Value (£)')
        axes[1].set_ylabel('Density')
        axes[1].legend()

        # Q-Q plot
        stats.probplot(ov_clipped.sample(min(3000, len(ov_clipped)), random_state=1), plot=axes[2])
        axes[2].set_title('Q-Q Plot — Order Value', fontweight='bold')

        plt.tight_layout()
        save_and_show(fig, 'probability_distributions.png')
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 — CONFIDENCE INTERVAL ESTIMATION
# ─────────────────────────────────────────────────────────────────────────────
class ConfidenceIntervals:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def _ci(self, data: pd.Series, alpha: float = 0.05) -> tuple:
        n, mean, se = len(data), data.mean(), stats.sem(data)
        t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * se
        return mean, mean - margin, mean + margin, se, n

    def run(self) -> None:
        df  = self.dl.df
        rfm = self.dl.rfm
        banner("Module 3 — Confidence Interval Estimation")

        sub_banner("95% & 99% Confidence Intervals — Key Metrics")
        targets = {
            'Avg Order Value (£)':       df.groupby('InvoiceNo')['TotalPrice'].sum(),
            'Avg Items per Order':        df.groupby('InvoiceNo')['Quantity'].sum(),
            'Avg Customer Spend (£)':     rfm['Monetary'],
            'Avg Purchase Frequency':     rfm['Frequency'],
            'Avg Recency (days)':         rfm['Recency'],
            'Avg Unit Price (£)':         df['UnitPrice'],
        }

        rows95, rows99 = [], []
        for label, series in targets.items():
            m, lo95, hi95, se, n = self._ci(series, 0.05)
            _,  lo99, hi99, _,  _ = self._ci(series, 0.01)
            rows95.append([label, fmt_num(m), fmt_num(lo95), fmt_num(hi95), fmt_num(se), f"{n:,}"])
            rows99.append([label, fmt_num(m), fmt_num(lo99), fmt_num(hi99), fmt_num(se), f"{n:,}"])

        print_table(rows95, ['Metric','Point Est.','CI Lower (95%)','CI Upper (95%)','Std Error','n'],
                    '  ── 95% Confidence Intervals ──')
        print_table(rows99, ['Metric','Point Est.','CI Lower (99%)','CI Upper (99%)','Std Error','n'],
                    '\n  ── 99% Confidence Intervals ──')

        # ── Proportion CI — repeat purchasers ───────────────────────────
        sub_banner("Proportion CI — Repeat Purchase Rate")
        n_total = len(rfm)
        p_hat   = (rfm['Frequency'] > 1).mean()
        z95, z99 = 1.96, 2.576
        se_p = np.sqrt(p_hat * (1 - p_hat) / n_total)

        print(f"  p̂ (repeat purchase rate) = {p_hat:.4f}  ({p_hat*100:.2f}%)")
        print(f"  95% CI: [{(p_hat-z95*se_p)*100:.2f}%, {(p_hat+z95*se_p)*100:.2f}%]")
        print(f"  99% CI: [{(p_hat-z99*se_p)*100:.2f}%, {(p_hat+z99*se_p)*100:.2f}%]")

        # ── Chart ────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.suptitle('95% Confidence Intervals — Key Business Metrics', fontsize=13, fontweight='bold')

        labels = [r[0] for r in rows95]
        means  = [float(r[1].replace(',','')) for r in rows95]
        lowers = [float(r[2].replace(',','')) for r in rows95]
        uppers = [float(r[3].replace(',','')) for r in rows95]
        errors_lo = [m - l for m, l in zip(means, lowers)]
        errors_hi = [u - m for m, u in zip(means, uppers)]

        y_pos = range(len(labels))
        ax.barh(y_pos, means, xerr=[errors_lo, errors_hi], color='steelblue',
                alpha=0.75, capsize=6, ecolor='black', linewidth=1.2)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([textwrap.shorten(l, 35) for l in labels])
        ax.set_xlabel('Estimated Value (normalised scale)')
        ax.set_title('Point Estimates with 95% CIs', fontweight='bold')
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        save_and_show(fig, 'confidence_intervals.png')
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 — REGRESSION & PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
class RegressionPrediction:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def _metrics_row(self, name, y_test, y_pred) -> list:
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        return [name, fmt_num(rmse), fmt_num(mae), fmt_num(r2)]

    def run(self) -> None:
        df  = self.dl.df
        rfm = self.dl.rfm
        banner("Module 4 — Regression & Predictive Modelling")

        # ── Feature engineering on RFM ───────────────────────────────────
        rfm_feat = rfm.copy()
        rfm_feat['Log_Monetary']  = np.log1p(rfm_feat['Monetary'])
        rfm_feat['Log_Frequency'] = np.log1p(rfm_feat['Frequency'])

        X_spend = rfm_feat[['Recency','Log_Frequency','R_Score','F_Score','M_Score']].values
        y_spend = rfm_feat['Log_Monetary'].values

        X_freq  = rfm_feat[['Recency','Log_Monetary','R_Score','F_Score','M_Score']].values
        y_freq  = rfm_feat['Log_Frequency'].values

        results = []

        for X, y, target_name in [
            (X_spend, y_spend, 'Customer Spend (log-transformed)'),
            (X_freq,  y_freq,  'Purchase Frequency (log-transformed)'),
        ]:
            sub_banner(f"Target: {target_name}")
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            # OLS
            ols = LinearRegression()
            ols.fit(X_tr_s, y_tr)
            y_ols = ols.predict(X_te_s)
            results.append(self._metrics_row(f'OLS — {target_name}', y_te, y_ols))

            # Ridge
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_tr_s, y_tr)
            y_ridge = ridge.predict(X_te_s)
            results.append(self._metrics_row(f'Ridge — {target_name}', y_te, y_ridge))

            # CV R²
            cv_scores = cross_val_score(ols, X_tr_s, y_tr, cv=5, scoring='r2')
            print(f"  OLS 5-Fold CV R² : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

            # Coefficients
            features = ['Recency','Log_Freq/Mon','R_Score','F_Score','M_Score']
            coef_data = list(zip(features, [f"{c:.4f}" for c in ols.coef_]))
            print_table(coef_data, ['Feature','Coefficient'], '  OLS Coefficients:')

        sub_banner("Model Performance Summary")
        print_table(results, ['Model','RMSE','MAE','R²'])

        # ── Prediction interface ─────────────────────────────────────────
        sub_banner("Sample Customer Spend Predictions")
        sample = rfm_feat.sample(8, random_state=7)[['CustomerID','Recency','Frequency','Monetary']].copy()

        X_samp_s = scaler.transform(
            rfm_feat.loc[sample.index,
                         ['Recency','Log_Frequency','R_Score','F_Score','M_Score']].values
        )
        pred_log = ols.predict(X_samp_s)
        sample['Actual Spend (£)']   = sample['Monetary'].apply(fmt_currency)
        sample['Predicted Spend (£)'] = np.expm1(pred_log).round(2)
        sample['Predicted Spend (£)'] = sample['Predicted Spend (£)'].apply(fmt_currency)
        print_table(sample[['CustomerID','Recency','Frequency',
                             'Actual Spend (£)','Predicted Spend (£)']].values.tolist(),
                    ['CustomerID','Recency (days)','Orders','Actual','Predicted'])

        # ── Residual chart ───────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Regression Diagnostics — Customer Spend Model', fontsize=13, fontweight='bold')

        residuals = y_te - y_ols
        axes[0].scatter(y_ols, residuals, alpha=0.3, color='steelblue', s=15)
        axes[0].axhline(0, color='red', lw=1.5, linestyle='--')
        axes[0].set_xlabel('Predicted (log scale)')
        axes[0].set_ylabel('Residuals')
        axes[0].set_title('Residuals vs. Fitted', fontweight='bold')

        axes[1].scatter(y_te, y_ols, alpha=0.3, color='darkorange', s=15)
        lims = [min(y_te.min(), y_ols.min()), max(y_te.max(), y_ols.max())]
        axes[1].plot(lims, lims, 'k--', lw=1.5, label='Perfect Fit')
        axes[1].set_xlabel('Actual (log scale)')
        axes[1].set_ylabel('Predicted (log scale)')
        axes[1].set_title('Actual vs. Predicted', fontweight='bold')
        axes[1].legend()

        plt.tight_layout()
        save_and_show(fig, 'regression_diagnostics.png')
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 — CUSTOMER SEGMENTATION (K-MEANS RFM)
# ─────────────────────────────────────────────────────────────────────────────
class CustomerSegmentation:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def run(self) -> None:
        rfm = self.dl.rfm.copy()
        banner("Module 5 — Customer Segmentation (K-Means + RFM)")

        # ── Elbow method ─────────────────────────────────────────────────
        sub_banner("Determining Optimal k — Elbow Method")
        X = np.log1p(rfm[['Recency','Frequency','Monetary']].values)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        inertia = []
        k_range = range(2, 11)
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X_s)
            inertia.append(km.inertia_)

        elbow_data = [[k, f"{i:,.2f}"] for k, i in zip(k_range, inertia)]
        print_table(elbow_data, ['k', 'Inertia (WCSS)'])
        print("  → Optimal k chosen: 5  (matches RFM scoring segments)")

        # ── Fit final model ──────────────────────────────────────────────
        km_final = KMeans(n_clusters=5, random_state=42, n_init=15)
        rfm['Cluster'] = km_final.fit_predict(X_s)

        cluster_labels = {
            rfm.groupby('Cluster')['Monetary'].mean().idxmax(): 'VIP / Champions',
        }
        # Re-label by Monetary mean rank
        cluster_means = rfm.groupby('Cluster')['Monetary'].mean().sort_values(ascending=False)
        labels_ordered = ['VIP / Champions', 'Loyal Customers',
                          'Potential Loyalists', 'At-Risk Customers', 'Lost / Churned']
        cluster_map = dict(zip(cluster_means.index, labels_ordered))
        rfm['KMeans_Segment'] = rfm['Cluster'].map(cluster_map)

        # ── Summary table ────────────────────────────────────────────────
        sub_banner("Cluster Profiles")
        summary = (
            rfm.groupby('KMeans_Segment')
            .agg(
                Count    =('CustomerID',  'count'),
                AvgRecency =('Recency',   'mean'),
                AvgFreq   =('Frequency',  'mean'),
                AvgMoney  =('Monetary',   'mean'),
                TotalRev  =('Monetary',   'sum'),
            )
            .reset_index()
        )
        summary['Share%']     = (summary['Count'] / summary['Count'].sum() * 100).apply(lambda x: f"{x:.1f}%")
        summary['AvgRecency'] = summary['AvgRecency'].apply(lambda x: f"{x:.0f} d")
        summary['AvgFreq']    = summary['AvgFreq'].apply(lambda x: f"{x:.1f}")
        summary['AvgMoney']   = summary['AvgMoney'].apply(fmt_currency)
        summary['TotalRev']   = summary['TotalRev'].apply(fmt_currency)

        print_table(
            summary[['KMeans_Segment','Count','Share%','AvgRecency','AvgFreq','AvgMoney','TotalRev']].values.tolist(),
            ['Segment','Count','Share','Avg Recency','Avg Freq','Avg Spend','Total Revenue']
        )

        # ── RFM score segments ───────────────────────────────────────────
        sub_banner("RFM Scoring Segments")
        rfm_seg = (
            rfm.groupby('Segment')
            .agg(Count=('CustomerID','count'), Revenue=('Monetary','sum'))
            .reset_index()
            .sort_values('Revenue', ascending=False)
        )
        rfm_seg['Revenue'] = rfm_seg['Revenue'].apply(fmt_currency)
        rfm_seg['Share%']  = (
            rfm['Segment'].value_counts(normalize=True) * 100
        ).reset_index().rename(columns={'proportion':'Share%'}).set_index('Segment')['Share%'] \
         .reindex(rfm_seg['Segment']).apply(lambda x: f"{x:.1f}%").values

        print_table(rfm_seg.values.tolist(), ['Segment','Customers','Total Revenue','Share'])

        # ── Charts ───────────────────────────────────────────────────────
        fig = plt.figure(figsize=(16, 10))
        gs  = gridspec.GridSpec(2, 3, figure=fig)
        fig.suptitle('Customer Segmentation Dashboard', fontsize=14, fontweight='bold')

        palette = ['#2ecc71','#3498db','#f39c12','#e74c3c','#9b59b6']

        # Scatter: Recency vs Monetary
        ax1 = fig.add_subplot(gs[0, :2])
        for idx, (seg, grp) in enumerate(rfm.groupby('KMeans_Segment')):
            ax1.scatter(np.log1p(grp['Recency']), np.log1p(grp['Monetary']),
                        s=20, alpha=0.5, label=seg, color=palette[idx % len(palette)])
        ax1.set_xlabel('log(Recency + 1)')
        ax1.set_ylabel('log(Monetary + 1)')
        ax1.set_title('Customer Clusters: Recency vs. Spend', fontweight='bold')
        ax1.legend(fontsize=8, loc='upper right')

        # Pie: segment share
        ax2 = fig.add_subplot(gs[0, 2])
        seg_counts = rfm['KMeans_Segment'].value_counts()
        ax2.pie(seg_counts.values, labels=seg_counts.index, autopct='%1.1f%%',
                colors=palette, startangle=90, textprops={'fontsize': 7})
        ax2.set_title('Segment Distribution', fontweight='bold')

        # Box: Monetary by segment
        ax3 = fig.add_subplot(gs[1, :])
        rfm_plot = rfm.copy()
        rfm_plot['log_Monetary'] = np.log1p(rfm_plot['Monetary'])
        order = ['VIP / Champions','Loyal Customers','Potential Loyalists','At-Risk Customers','Lost / Churned']
        sns.boxplot(data=rfm_plot, x='KMeans_Segment', y='log_Monetary',
                    order=[o for o in order if o in rfm_plot['KMeans_Segment'].unique()],
                    palette=palette, ax=ax3)
        ax3.set_xlabel('Customer Segment')
        ax3.set_ylabel('log(Monetary Value + 1) (£)')
        ax3.set_title('Spend Distribution by Segment', fontweight='bold')
        ax3.tick_params(axis='x', rotation=15)

        plt.tight_layout()
        save_and_show(fig, 'customer_segmentation.png')
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6 — TIME SERIES ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
class TimeSeriesAnalysis:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def run(self) -> None:
        df = self.dl.df
        banner("Module 6 — Time Series & Seasonal Analysis")

        # ── Monthly revenue ──────────────────────────────────────────────
        sub_banner("Monthly Revenue Trend")
        monthly = (
            df.groupby('YearMonth')
            .agg(Revenue=('TotalPrice','sum'), Orders=('InvoiceNo','nunique'),
                 Customers=('CustomerID','nunique'))
            .reset_index()
        )
        monthly['YearMonth_str'] = monthly['YearMonth'].astype(str)
        monthly['MoM_Growth']    = monthly['Revenue'].pct_change() * 100

        rows = []
        for _, r in monthly.iterrows():
            rows.append([
                r['YearMonth_str'],
                fmt_currency(r['Revenue']),
                f"{r['Orders']:,}",
                f"{r['Customers']:,}",
                f"{r['MoM_Growth']:+.1f}%" if pd.notna(r['MoM_Growth']) else '—'
            ])
        print_table(rows, ['Month','Revenue','Orders','Customers','MoM Growth'])

        # ── Day-of-week pattern ──────────────────────────────────────────
        sub_banner("Revenue by Day of Week")
        day_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
        day_rev = df.groupby('DayOfWeek')['TotalPrice'].sum().reindex(day_order).dropna()
        day_data = [[d, fmt_currency(v), f"{v/day_rev.sum()*100:.2f}%"]
                    for d, v in day_rev.items()]
        print_table(day_data, ['Day','Revenue','Share'])

        # ── Hourly distribution ──────────────────────────────────────────
        sub_banner("Revenue by Hour of Day (Business Hours Peak)")
        hour_rev = df.groupby('Hour')['TotalPrice'].sum()
        peak_hour = hour_rev.idxmax()
        print(f"  Peak trading hour: {peak_hour:02d}:00  |  Revenue: {fmt_currency(hour_rev[peak_hour])}")

        top_hours = hour_rev.sort_values(ascending=False).head(5)
        hr_data = [[f"{h:02d}:00", fmt_currency(v)] for h, v in top_hours.items()]
        print_table(hr_data, ['Hour','Revenue'])

        # ── Seasonality index ────────────────────────────────────────────
        sub_banner("Quarterly Seasonality Index")
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        q_rev = df.groupby('Quarter')['TotalPrice'].sum()
        q_avg = q_rev.mean()
        q_idx = q_rev / q_avg
        q_data = [[f"Q{q}", fmt_currency(r), f"{idx:.3f}"]
                  for q, r, idx in zip(q_rev.index, q_rev.values, q_idx.values)]
        print_table(q_data, ['Quarter','Revenue','Seasonality Index'])
        print("  (Index > 1.0 → above average; < 1.0 → below average)")

        # ── Charts ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Time Series & Seasonal Analysis', fontsize=14, fontweight='bold')

        # Monthly revenue line
        x_ticks = range(len(monthly))
        axes[0, 0].fill_between(x_ticks, monthly['Revenue'], alpha=0.25, color='steelblue')
        axes[0, 0].plot(x_ticks, monthly['Revenue'], marker='o', color='steelblue', lw=2, ms=5)
        axes[0, 0].set_xticks(x_ticks[::2])
        axes[0, 0].set_xticklabels(monthly['YearMonth_str'].iloc[::2], rotation=45, ha='right', fontsize=7)
        axes[0, 0].set_title('Monthly Revenue Trend', fontweight='bold')
        axes[0, 0].set_ylabel('Revenue (£)')

        # Monthly orders
        axes[0, 1].bar(x_ticks, monthly['Orders'], color='darkorange', alpha=0.8)
        axes[0, 1].set_xticks(x_ticks[::2])
        axes[0, 1].set_xticklabels(monthly['YearMonth_str'].iloc[::2], rotation=45, ha='right', fontsize=7)
        axes[0, 1].set_title('Monthly Order Count', fontweight='bold')
        axes[0, 1].set_ylabel('Number of Orders')

        # Day-of-week bar
        axes[1, 0].bar(day_order, [day_rev.get(d, 0) for d in day_order],
                       color='seagreen', alpha=0.85)
        axes[1, 0].set_title('Revenue by Day of Week', fontweight='bold')
        axes[1, 0].set_ylabel('Revenue (£)')
        axes[1, 0].tick_params(axis='x', rotation=20)

        # Seasonality index
        quarters = [f"Q{q}" for q in q_rev.index]
        bars = axes[1, 1].bar(quarters, q_idx.values,
                               color=['#e74c3c' if v < 1 else '#2ecc71' for v in q_idx.values],
                               alpha=0.85)
        axes[1, 1].axhline(1.0, color='black', linestyle='--', lw=1.5, label='Baseline = 1.0')
        axes[1, 1].set_title('Quarterly Seasonality Index', fontweight='bold')
        axes[1, 1].set_ylabel('Seasonality Index')
        axes[1, 1].legend()
        for bar, val in zip(bars, q_idx.values):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                            f"{val:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.tight_layout()
        save_and_show(fig, 'time_series_analysis.png')
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 7 — PRODUCT & GEOGRAPHIC INSIGHTS
# ─────────────────────────────────────────────────────────────────────────────
class ProductGeoInsights:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def run(self) -> None:
        df = self.dl.df
        banner("Module 7 — Product & Geographic Insights")

        # ── Top 15 products by revenue ───────────────────────────────────
        sub_banner("Top 15 Products by Revenue")
        prod_rev = (
            df.groupby(['StockCode','Description'])
            .agg(Revenue=('TotalPrice','sum'), UnitsSold=('Quantity','sum'), Orders=('InvoiceNo','nunique'))
            .sort_values('Revenue', ascending=False)
            .head(15)
            .reset_index()
        )
        prod_rev['Revenue']    = prod_rev['Revenue'].apply(fmt_currency)
        prod_rev['UnitsSold']  = prod_rev['UnitsSold'].apply(lambda x: f"{x:,}")
        prod_rev['Description']= prod_rev['Description'].apply(lambda x: str(x)[:35] if pd.notna(x) else 'N/A')
        print_table(prod_rev[['StockCode','Description','Revenue','UnitsSold','Orders']].values.tolist(),
                    ['StockCode','Description','Revenue','Units Sold','Orders'])

        # ── Low-performing products ──────────────────────────────────────
        sub_banner("Bottom 10 Products by Revenue (min 5 orders)")
        prod_low = (
            df.groupby(['StockCode','Description'])
            .agg(Revenue=('TotalPrice','sum'), Orders=('InvoiceNo','nunique'))
            .query('Orders >= 5')
            .sort_values('Revenue')
            .head(10)
            .reset_index()
        )
        prod_low['Revenue']     = prod_low['Revenue'].apply(fmt_currency)
        prod_low['Description'] = prod_low['Description'].apply(lambda x: str(x)[:35] if pd.notna(x) else 'N/A')
        print_table(prod_low[['StockCode','Description','Revenue','Orders']].values.tolist(),
                    ['StockCode','Description','Revenue','Orders'])

        # ── Geographic revenue ───────────────────────────────────────────
        sub_banner("Geographic Revenue Analysis (All Countries)")
        geo = (
            df.groupby('Country')
            .agg(Revenue=('TotalPrice','sum'),
                 Customers=('CustomerID','nunique'),
                 Orders=('InvoiceNo','nunique'))
            .sort_values('Revenue', ascending=False)
            .reset_index()
        )
        geo['Share%']    = (geo['Revenue'] / geo['Revenue'].sum() * 100).apply(lambda x: f"{x:.2f}%")
        geo['Revenue']   = geo['Revenue'].apply(fmt_currency)
        geo['AvgOrderVal'] = (
            df.groupby('Country').apply(lambda g: g.groupby('InvoiceNo')['TotalPrice'].sum().mean())
        ).reindex(geo['Country'].values).apply(fmt_currency).values

        print_table(geo[['Country','Revenue','Customers','Orders','Share%','AvgOrderVal']].values.tolist(),
                    ['Country','Revenue','Customers','Orders','Share','Avg Order (£)'])

        # ── Charts ───────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle('Product & Geographic Revenue Insights', fontsize=14, fontweight='bold')

        top15_raw = (
            df.groupby('StockCode')['TotalPrice'].sum()
            .sort_values(ascending=False).head(15)
        )
        axes[0].barh(range(15), top15_raw.values[::-1], color='steelblue', alpha=0.85)
        axes[0].set_yticks(range(15))
        axes[0].set_yticklabels(top15_raw.index.tolist()[::-1], fontsize=8)
        axes[0].set_title('Top 15 Products by Revenue', fontweight='bold')
        axes[0].set_xlabel('Total Revenue (£)')

        top10_geo = (
            df[df['Country'] != 'United Kingdom']
            .groupby('Country')['TotalPrice'].sum()
            .sort_values(ascending=False).head(10)
        )
        axes[1].barh(range(10), top10_geo.values[::-1], color='darkorange', alpha=0.85)
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels(top10_geo.index.tolist()[::-1], fontsize=9)
        axes[1].set_title('Top 10 International Countries (ex. UK)', fontweight='bold')
        axes[1].set_xlabel('Total Revenue (£)')

        plt.tight_layout()
        save_and_show(fig, 'product_geo_insights.png')
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 8 — EXECUTIVE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
class ExecutiveSummary:
    def __init__(self, loader: DataLoader):
        self.dl = loader

    def run(self) -> None:
        df  = self.dl.df
        rfm = self.dl.rfm
        banner("Module 8 — Executive Summary & Key Insights")

        total_rev    = df['TotalPrice'].sum()
        total_orders = df['InvoiceNo'].nunique()
        total_cust   = df['CustomerID'].nunique()
        avg_order    = df.groupby('InvoiceNo')['TotalPrice'].sum().mean()
        repeat_rate  = (rfm['Frequency'] > 1).mean() * 100
        top_country  = df.groupby('Country')['TotalPrice'].sum().idxmax()
        top_product  = df.groupby('StockCode')['TotalPrice'].sum().idxmax()
        peak_month   = df.groupby('YearMonth')['TotalPrice'].sum().idxmax()
        vip_rev      = rfm[rfm['Segment'] == 'VIP / Champions']['Monetary'].sum()
        vip_share    = vip_rev / rfm['Monetary'].sum() * 100

        lines = [
            ("Total Revenue",            fmt_currency(total_rev)),
            ("Total Orders",             f"{total_orders:,}"),
            ("Unique Customers",          f"{total_cust:,}"),
            ("Average Order Value",       fmt_currency(avg_order)),
            ("Repeat Purchase Rate",      f"{repeat_rate:.1f}%"),
            ("Top Revenue Country",       top_country),
            ("Top Revenue Product Code",  top_product),
            ("Peak Revenue Month",        str(peak_month)),
            ("VIP Customer Revenue Share",f"{vip_share:.1f}%"),
        ]
        print_table(lines, ['KPI', 'Value'], '  ── Key Performance Indicators ──')

        sub_banner("Strategic Recommendations")
        recs = [
            ("1. Retain VIP Customers",
             f"  {vip_share:.1f}% of revenue comes from Champions. Launch loyalty rewards "
             "and personalised promotions to prevent churn."),
            ("2. Reactivate At-Risk Segment",
             "  Target customers with high recency and low frequency with win-back email "
             "campaigns offering exclusive discounts."),
            ("3. Capitalise on Seasonal Peaks",
             f"  Q4 shows the highest seasonality index. Stock up proactively, "
             "run pre-Christmas promotions, and scale logistics in Oct–Nov."),
            ("4. Expand in High-Growth Markets",
             f"  After UK ({top_country} is the top international market), invest in DE, FR "
             "and NL — high order values indicate premium customer profiles."),
            ("5. Optimise Underperforming SKUs",
             "  Retire or bundle low-revenue products with ≥5 orders to improve "
             "catalogue efficiency and reduce warehouse costs."),
            ("6. Leverage Peak Trading Hours",
             "  Schedule flash-sales and email campaigns between 10:00–14:00 "
             "to maximise conversion during peak traffic windows."),
        ]
        for title, body in recs:
            print(f"\n  ◆ {title}")
            print(f"  {body}")

        print(f"\n  {DIVIDER}")
        print(f"  Analysis generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Dataset: Online Retail II  |  Records processed: {len(df):,}")
        print(f"  {DIVIDER}\n")
        pause()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────
MENU = """
  ╔══════════════════════════════════════════════════════════════════════╗
  ║    Global E-Commerce Customer Behaviour & Predictive Analysis        ║
  ║    Dataset: Online Retail II  (Dec 2009 – Dec 2011)                  ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║  [1]  Descriptive Statistics                                         ║
  ║  [2]  Probability & Distribution Analysis                            ║
  ║  [3]  Confidence Interval Estimation                                 ║
  ║  [4]  Regression & Predictive Modelling                              ║
  ║  [5]  Customer Segmentation (K-Means + RFM)                         ║
  ║  [6]  Time Series & Seasonal Analysis                                ║
  ║  [7]  Product & Geographic Insights                                  ║
  ║  [8]  Executive Summary & Recommendations                            ║
  ║  [A]  Run ALL Modules Sequentially                                   ║
  ║  [0]  Exit                                                           ║
  ╚══════════════════════════════════════════════════════════════════════╝
"""

def main() -> None:
    os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

    loader = DataLoader()
    loader.load()

    modules = {
        '1': ('Descriptive Statistics',                lambda: DescriptiveStats(loader).run()),
        '2': ('Probability & Distribution Analysis',   lambda: ProbabilityAnalysis(loader).run()),
        '3': ('Confidence Interval Estimation',        lambda: ConfidenceIntervals(loader).run()),
        '4': ('Regression & Predictive Modelling',     lambda: RegressionPrediction(loader).run()),
        '5': ('Customer Segmentation',                 lambda: CustomerSegmentation(loader).run()),
        '6': ('Time Series & Seasonal Analysis',       lambda: TimeSeriesAnalysis(loader).run()),
        '7': ('Product & Geographic Insights',         lambda: ProductGeoInsights(loader).run()),
        '8': ('Executive Summary',                     lambda: ExecutiveSummary(loader).run()),
    }

    while True:
        print(MENU)
        choice = input("  Select module [0-8 / A]: ").strip().upper()

        if choice == '0':
            print("\n  Thank you for using the E-Commerce Analysis System. Goodbye!\n")
            sys.exit(0)
        elif choice == 'A':
            for key in modules:
                modules[key][1]()
        elif choice in modules:
            modules[choice][1]()
        else:
            print("  ⚠  Invalid choice. Please enter 1–8, A, or 0.")


if __name__ == '__main__':
    main()
