"""
Decomposing Implied Volatility Spread:
Separating Underlying Microstructure Effects from True Volatility
in Indian Options Markets (BANKNIFTY)

Based on the thesis by Shriraj Aher.
This script reads from options-data.csv (NSE option chain export format).
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")


# =========================================
# CONFIGURATION
# =========================================

CSV_PATH      = "options-data.csv"  # Path to your options chain CSV
SPOT          = 56500.0             # Approximate BANKNIFTY spot price
T_DAYS        = 2                   # Days to expiry (short-dated, as per thesis)
RISK_FREE     = 0.067               # Risk-free rate (Indian T-bill ~6.7%)
SPREAD_BPS    = 0.0005              # Underlying bid-ask spread (0.05 bps)
MONEYNESS_LO  = 0.94                # Lower bound of strike filter (% of spot)
MONEYNESS_HI  = 1.06                # Upper bound of strike filter (% of spot)


# =========================================
# 1. LOAD AND CLEAN DATA
# =========================================

def load_option_chain(csv_path: str) -> pd.DataFrame:
    """
    Load NSE option chain CSV. The NSE format has:
      - Row 0: 'CALLS,,PUTS' header
      - Row 1: Column names (OI, CHNG IN OI, ..., STRIKE, ..., OI)
      - Rows 2+: Data with comma-formatted numbers
    Returns a clean DataFrame with CALLS columns: strike, bid, ask, lastPrice
    """
    df = pd.read_csv(csv_path, skiprows=1, header=0, on_bad_lines="skip")

    # CALLS side columns: STRIKE, BID, ASK, LTP
    calls = df[["STRIKE", "BID", "ASK", "LTP"]].copy()
    calls.columns = ["strike", "bid", "ask", "lastPrice"]

    def to_float(val):
        try:
            return float(str(val).replace(",", "").strip())
        except (ValueError, AttributeError):
            return np.nan

    for col in calls.columns:
        calls[col] = calls[col].apply(to_float)

    # Drop rows with missing or zero bid/ask
    calls = calls.dropna(subset=["strike", "bid", "ask"])
    calls = calls[(calls["bid"] > 0) & (calls["ask"] > 0)].reset_index(drop=True)

    # Use mid-price as lastPrice where lastPrice is missing or suspicious
    calls["mid"] = (calls["bid"] + calls["ask"]) / 2
    mask = calls["lastPrice"].isna() | (calls["lastPrice"] <= 0)
    calls.loc[mask, "lastPrice"] = calls.loc[mask, "mid"]

    return calls


# =========================================
# 2. BLACK-SCHOLES PRICER
# =========================================

def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if sigma <= 1e-9 or T <= 1e-9:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes Gamma: d²C/dS²"""
    if sigma <= 1e-9 or T <= 1e-9:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def implied_vol(price: float, S: float, K: float, T: float, r: float) -> float:
    """
    Solve for implied volatility using Brent's method.
    Returns NaN if price <= intrinsic or solver fails.
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic + 1e-4:
        return np.nan
    try:
        return brentq(
            lambda sigma: bs_call(S, K, T, r, sigma) - price,
            1e-6, 10.0,
            xtol=1e-8, maxiter=200
        )
    except (ValueError, RuntimeError):
        return np.nan


# =========================================
# 3. SPREAD DECOMPOSITION
# =========================================

def decompose_iv_spread(df: pd.DataFrame, spot: float, T: float,
                        r: float, spread_bps: float) -> pd.DataFrame:
    """
    For each strike, compute:
      - iv_naive  : IV(mid_price, S_mid)  — naive, ignores underlying spread
      - iv_bid    : IV(bid_price, S_bid)  — adjusted for underlying bid
      - iv_ask    : IV(ask_price, S_ask)  — adjusted for underlying ask
      - delta_obs : Naive IV spread  = IV(ask, S_mid) - IV(bid, S_mid)
      - delta_adj : Adjusted IV spread = iv_ask - iv_bid
      - gamma     : Option gamma at mid-vol (sensitivity indicator)
    """
    S_mid = spot
    S_bid = spot * (1 - spread_bps)
    S_ask = spot * (1 + spread_bps)

    records = []
    for _, row in df.iterrows():
        K          = row["strike"]
        mid_price  = row["lastPrice"]
        bid_price  = row["bid"]
        ask_price  = row["ask"]

        # Naive IV: use mid underlying for all prices
        iv_naive_bid = implied_vol(bid_price, S_mid, K, T, r)
        iv_naive_ask = implied_vol(ask_price, S_mid, K, T, r)
        iv_naive     = implied_vol(mid_price, S_mid, K, T, r)

        # Adjusted IV: use bid underlying for bid price, ask underlying for ask price
        iv_bid = implied_vol(bid_price, S_bid, K, T, r)
        iv_ask = implied_vol(ask_price, S_ask, K, T, r)

        # Gamma at naive IV (use fallback if NaN)
        sigma_for_gamma = iv_naive if not np.isnan(iv_naive) else 0.20
        gamma = bs_gamma(S_mid, K, T, r, sigma_for_gamma)

        records.append({
            "strike"      : K,
            "moneyness"   : K / S_mid,
            "bid"         : bid_price,
            "ask"         : ask_price,
            "mid_price"   : mid_price,
            "iv_naive"    : iv_naive,
            "iv_naive_bid": iv_naive_bid,
            "iv_naive_ask": iv_naive_ask,
            "iv_bid"      : iv_bid,
            "iv_ask"      : iv_ask,
            "delta_obs"   : iv_naive_ask - iv_naive_bid,   # Eq. 11 from thesis
            "delta_adj"   : iv_ask - iv_bid,               # Eq. 12 from thesis
            "gamma"       : gamma,
        })

    result = pd.DataFrame(records)

    # Drop rows where IV computation failed entirely
    result = result.dropna(subset=["iv_naive", "iv_bid", "iv_ask"])

    # Clip negative spreads to 0 (artefacts of numerical inversion)
    result["delta_obs"] = result["delta_obs"].clip(lower=0)
    result["delta_adj"] = result["delta_adj"].clip(lower=0)

    # Microstructure-explained fraction of IV spread
    micro_explained = 1 - (result["delta_adj"] / result["delta_obs"].replace(0, np.nan))
    result["micro_fraction"] = micro_explained.clip(0, 1)

    return result.reset_index(drop=True)


# =========================================
# 4. SUMMARY STATISTICS
# =========================================

def print_summary(res: pd.DataFrame, spot: float) -> None:
    print("\n" + "=" * 60)
    print("  IV SPREAD DECOMPOSITION — SUMMARY STATISTICS")
    print("=" * 60)
    print(f"  Spot (S_mid)            : {spot:,.2f}")
    print(f"  Strikes analysed        : {len(res)}")
    print(f"  Strike range            : {res['strike'].min():,.0f} – {res['strike'].max():,.0f}")
    print()

    obs = res["delta_obs"].dropna()
    adj = res["delta_adj"].dropna()
    mf  = res["micro_fraction"].dropna()

    print(f"  Mean naive IV spread    : {obs.mean():.4f}  ({obs.mean()*100:.2f}%)")
    print(f"  Mean adjusted IV spread : {adj.mean():.4f}  ({adj.mean()*100:.2f}%)")
    print(f"  Mean IV mid (smile)     : {res['iv_naive'].mean():.4f}  ({res['iv_naive'].mean()*100:.2f}%)")
    print()
    print(f"  Micro-structure explains on average {mf.mean()*100:.1f}% of IV spread")
    print(f"  (Min {mf.min()*100:.1f}% – Max {mf.max()*100:.1f}%)")
    print()

    # Hypothesis test: Eq. 13  Δσ_adj << Δσ_obs
    ratio = adj.mean() / obs.mean() if obs.mean() > 0 else np.nan
    print(f"  Δσ_adj / Δσ_obs ratio   : {ratio:.3f}")
    if ratio < 0.5:
        print("  ✓ CONFIRMED: Adjusted spread is << naive spread")
        print("    Underlying microstructure dominates IV spread")
    else:
        print("  ✗ NOT confirmed at this T; true vol uncertainty is significant")
    print("=" * 60 + "\n")


# =========================================
# 5. PLOTTING
# =========================================

def plot_results(res: pd.DataFrame, spot: float, T_days: int) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"IV Spread Decomposition — BANKNIFTY  |  Spot ≈ {spot:,.0f}  |  T = {T_days}d",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    x = res["strike"]

    # --- Plot 1: Volatility Smile (Naive IV) ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, res["iv_naive"] * 100, color="steelblue", lw=2, label="Naive IV (mid)")
    ax1.plot(x, res["iv_naive_bid"] * 100, color="green", lw=1.2,
             linestyle="--", label="Naive IV (bid)")
    ax1.plot(x, res["iv_naive_ask"] * 100, color="red", lw=1.2,
             linestyle="--", label="Naive IV (ask)")
    ax1.axvline(spot, color="black", linestyle=":", alpha=0.6, label=f"Spot {spot:,.0f}")
    ax1.set_title("Volatility Smile — Naive IVs", fontsize=11)
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Implied Volatility (%)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Adjusted IV (Thesis §5.2) ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, res["iv_naive"] * 100, color="steelblue", lw=2, label="Naive IV (mid)")
    ax2.plot(x, res["iv_bid"] * 100, color="green", lw=1.5,
             linestyle="--", label="Adjusted IV (bid S)")
    ax2.plot(x, res["iv_ask"] * 100, color="red", lw=1.5,
             linestyle="--", label="Adjusted IV (ask S)")
    ax2.axvline(spot, color="black", linestyle=":", alpha=0.6)
    ax2.set_title("Volatility Smile — Adjusted IVs\n(Underlying spread accounted for)",
                  fontsize=11)
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Implied Volatility (%)")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Spread Comparison (Thesis §5.3) ---
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.fill_between(x, res["delta_obs"] * 100, alpha=0.35,
                     color="tomato", label="Δσ_obs (Naive spread)")
    ax3.fill_between(x, res["delta_adj"] * 100, alpha=0.50,
                     color="mediumseagreen", label="Δσ_adj (Adjusted spread)")
    ax3.plot(x, res["delta_obs"] * 100, color="tomato", lw=1.5)
    ax3.plot(x, res["delta_adj"] * 100, color="mediumseagreen", lw=1.5)
    ax3.axvline(spot, color="black", linestyle=":", alpha=0.6)
    ax3.set_title("IV Spread Decomposition\nΔσ_obs vs Δσ_adj", fontsize=11)
    ax3.set_xlabel("Strike")
    ax3.set_ylabel("IV Spread (%)")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Microstructure-Explained Fraction & Gamma ---
    ax4 = fig.add_subplot(gs[1, 1])
    color_bar = "darkorchid"
    color_line = "darkorange"
    bars = ax4.bar(x, res["micro_fraction"] * 100, width=(x.max() - x.min()) / len(x) * 0.8,
                   color=color_bar, alpha=0.55, label="Micro-structure fraction (%)")
    ax4.set_ylabel("Micro-structure Explained (%)", color=color_bar)
    ax4.tick_params(axis="y", labelcolor=color_bar)

    ax4b = ax4.twinx()
    ax4b.plot(x, res["gamma"], color=color_line, lw=2, label="Gamma")
    ax4b.set_ylabel("Gamma", color=color_line)
    ax4b.tick_params(axis="y", labelcolor=color_line)

    ax4.axvline(spot, color="black", linestyle=":", alpha=0.6)
    ax4.set_title("Micro-structure Effect & Gamma\n(Short-dated options → high gamma)", fontsize=11)
    ax4.set_xlabel("Strike")
    ax4.set_ylim(0, 110)

    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax4.grid(True, alpha=0.3)

    plt.savefig("IV_decomposition.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Plot saved to: IV_decomposition.png")


# =========================================
# 6. MAIN
# =========================================

def main():
    print("\nLoading option chain data from:", CSV_PATH)
    raw = load_option_chain(CSV_PATH)
    print(f"  Total strikes with valid bid/ask: {len(raw)}")

    # Filter to near-money strikes
    lo = SPOT * MONEYNESS_LO
    hi = SPOT * MONEYNESS_HI
    df = raw[(raw["strike"] >= lo) & (raw["strike"] <= hi)].copy()
    print(f"  Near-money strikes ({MONEYNESS_LO:.0%}–{MONEYNESS_HI:.0%} of spot): {len(df)}")

    # Time to expiry
    T = max(T_DAYS / 365.0, 1e-5)
    print(f"  T = {T_DAYS} days = {T:.6f} years")

    # Decompose spreads
    print("\nComputing implied volatilities...")
    res = decompose_iv_spread(df, SPOT, T, RISK_FREE, SPREAD_BPS)
    print(f"  Valid IV rows after computation: {len(res)}")

    # Summary
    print_summary(res, SPOT)

    # Save result table
    out_cols = ["strike", "moneyness", "bid", "ask", "mid_price",
                "iv_naive", "iv_bid", "iv_ask",
                "delta_obs", "delta_adj", "micro_fraction", "gamma"]
    res[out_cols].round(6).to_csv("IV_decomposition_results.csv", index=False)
    print("Results saved to: IV_decomposition_results.csv")

    # Plot
    plot_results(res, SPOT, T_DAYS)


if __name__ == "__main__":
    main()