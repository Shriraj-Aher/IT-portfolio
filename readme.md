# Decomposing Implied Volatility Spread

### Separating Underlying Microstructure Effects from True Volatility in Indian Options Markets

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Market](https://img.shields.io/badge/Market-NSE%20India-orange?style=flat-square)](https://www.nseindia.com/)
[![Instrument](https://img.shields.io/badge/Instrument-BANKNIFTY%20Options-red?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-Research-purple?style=flat-square)]()

---

## Abstract

> Options traders routinely read implied volatility (IV) spreads as signals of volatility uncertainty. This project challenges that interpretation.
>
> Using **BANKNIFTY weekly index options** near expiry, we construct a framework to decompose the observed bid–ask spread in implied volatility into two components: a portion attributable to **underlying asset microstructure** (the bid–ask spread in the spot price itself), and a residual reflecting **genuine volatility dispersion**. For short-dated, high-gamma options, we find that **underlying microstructure explains ~96% of the apparent IV spread** — suggesting that most of what traders observe as "volatility uncertainty" is, in fact, a mechanical artifact of delta-one price noise amplified through gamma.

---

## Table of Contents

- [Motivation](#motivation)
- [Theoretical Framework](#theoretical-framework)
- [Methodology](#methodology)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Limitations & Extensions](#limitations--extensions)
- [References](#references)

---

## Motivation

At the point of trade, most Black–Scholes inputs are fixed:

| Parameter | Status at trade time |
|-----------|---------------------|
| Option type (call/put) | ✅ Fixed |
| Strike price *K* | ✅ Fixed |
| Time to maturity *T* | ✅ Fixed |
| Risk-free rate *r* | ✅ Essentially fixed |
| **Underlying price *S*** | ⚠️ Quoted as a spread |
| **Volatility *σ*** | ⚠️ Implied, not observed |

The key insight is that because *S* itself is quoted with a bid–ask spread, **any IV computed from an option price implicitly inherits noise from the underlying spread**. This effect is amplified by **gamma** (∂²C/∂S²), which grows sharply as expiry approaches.

The practical implication: an IV spread that looks like a 1.5% volatility disagreement may actually reflect only a ~0.28% true volatility difference, with the remainder being a pure microstructure artifact.

---

## Theoretical Framework

### Black–Scholes Pricing

```
C = S·Φ(d₁) − K·e^(−rT)·Φ(d₂)

       ln(S/K) + (r + σ²/2)·T
d₁ = ─────────────────────────
              σ·√T

d₂ = d₁ − σ·√T
```

### Implied Volatility

IV is defined implicitly as the solution *σ* to:

```
C_market = C_BS(S, K, T, r, σ)
```

Solved numerically via Brent's method.

### Spread Decomposition

Let the market quote two underlying prices (S_bid, S_ask) and two option prices (C_bid, C_ask).

**Naive IV spread** (ignores underlying microstructure):

```
Δσ_obs = IV(C_ask, S_mid) − IV(C_bid, S_mid)       [Eq. 11]
```

**Adjusted IV spread** (accounts for underlying bid–ask):

```
Δσ_adj = IV(C_ask, S_ask) − IV(C_bid, S_bid)       [Eq. 12]
```

**Core hypothesis**:

```
Δσ_adj ≪ Δσ_obs                                     [Eq. 13]
```

### Gamma as the Amplification Channel

```
Γ = ∂²C/∂S² = φ(d₁) / (S·σ·√T)
```

As *T* → 0, Γ → ∞ near the money. A 0.05% move in *S* can produce a large move in *C*, which is then back-solved into a large apparent shift in IV.

---

## Methodology

```
┌─────────────────────────────────────────────────────────┐
│                  NSE Option Chain CSV                   │
│         (BANKNIFTY weekly near-expiry options)          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
              ┌─────────────────┐
              │  Data Cleaning  │  Strip commas, fill mid-price,
              │  & Filtering    │  restrict to near-money strikes
              └────────┬────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
    IV(C_mid,    IV(C_bid,    IV(C_ask,
      S_mid)       S_bid)       S_ask)
          │            │            │
          └────────────┼────────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Spread Decomposition │
           │  Δσ_obs vs Δσ_adj    │
           └───────────┬───────────┘
                       │
                       ▼
           ┌───────────────────────┐
           │  Microstructure       │
           │  Fraction & Gamma     │
           └───────────────────────┘
```

**Assumptions:**

- Black–Scholes framework (constant vol within the inversion)
- Underlying bid–ask spread proxy: ±0.05 bps of mid-price
- Risk-free rate: 6.7% (Indian 91-day T-bill)
- Focus on short-dated options: T ≤ 2 days (weekly expiry)

---

## Repository Structure

```
iv-spread-decomposition/
│
├── IV_project_final.py              # Main analysis script
├── options-data.csv                 # NSE BANKNIFTY option chain data
├── IV_decomposition_results.csv     # Per-strike output table
├── IV_decomposition.png             # Output plots (4-panel figure)
├── IV_project.pdf                   # Research paper / thesis
└── README.md                        # This file
```

---

## Installation

**Requirements:** Python 3.9+

```bash
# Clone the repository
git clone https://github.com/<your-username>/iv-spread-decomposition.git
cd iv-spread-decomposition

# Install dependencies
pip install numpy pandas scipy matplotlib
```

No exotic dependencies. No live data feed required — the CSV is self-contained.

---

## Usage

```bash
python IV_project_final.py
```

**Configuration** — edit the constants at the top of `IV_project_final.py`:

```python
CSV_PATH      = "options-data.csv"   # Path to NSE option chain export
SPOT          = 56500.0              # BANKNIFTY spot price at snapshot time
T_DAYS        = 2                    # Days to expiry
RISK_FREE     = 0.067                # Risk-free rate
SPREAD_BPS    = 0.0005               # Underlying bid-ask half-spread
MONEYNESS_LO  = 0.94                 # Lower strike filter (% of spot)
MONEYNESS_HI  = 1.06                 # Upper strike filter (% of spot)
```

**Output:**

| File | Description |
|------|-------------|
| `IV_decomposition.png` | 4-panel figure: smile, adjusted IVs, spread comparison, gamma |
| `IV_decomposition_results.csv` | Per-strike table with all IV and spread metrics |
| Console | Summary statistics including Δσ_adj/Δσ_obs ratio |

---

## Results

Results from the BANKNIFTY snapshot (T = 2 days, spot ≈ 56,500):

| Metric | Value |
|--------|-------|
| Strikes analysed | 67 |
| Strike range | 53,200 – 59,800 |
| Mean naive IV spread (Δσ_obs) | **1.57%** |
| Mean adjusted IV spread (Δσ_adj) | **0.28%** |
| Δσ_adj / Δσ_obs ratio | **0.180** |
| Microstructure-explained fraction | **~96.2%** |

**Interpretation:** The adjusted IV spread collapses to ~18% of the naive spread. The apparent volatility disagreement between market makers is not primarily a disagreement about volatility — it is an amplified echo of the underlying bid–ask spread, channeled through the option's gamma.

### Output Plots

The script generates a 4-panel figure:

| Panel | Description |
|-------|-------------|
| Top-left | Volatility smile — naive IVs (mid, bid, ask) |
| Top-right | Volatility smile — adjusted IVs with underlying spread correction |
| Bottom-left | IV spread comparison: Δσ_obs vs Δσ_adj |
| Bottom-right | Microstructure-explained fraction (%) alongside gamma |

---

## Limitations & Extensions

**Current Limitations:**

- Single snapshot in time; no time-series analysis of spread dynamics
- Underlying spread proxied by a fixed ±0.05 bps rather than real L2 order book data
- Black–Scholes inversion assumes flat vol; a local vol or SABR inversion would be more rigorous
- Transaction costs and market impact not modelled
- Puts not included (put–call parity would allow a consistency check)

**Possible Extensions:**

- [ ] Intraday analysis: track how the microstructure fraction evolves through the trading session
- [ ] Cross-asset comparison: NIFTY vs BANKNIFTY — does higher underlying liquidity reduce the fraction?
- [ ] SABR-based IV inversion to remove the constant-vol assumption
- [ ] True L2 data integration for a more accurate underlying spread estimate
- [ ] Greeks-based hedging cost model: translate Δσ_adj into P&L terms for market makers
- [ ] Regime analysis: does the microstructure fraction change around high-volatility events (RBI policy, elections)?

---

## References

1. Black, F., & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy, 81(3), 637–654.
2. Glosten, L. R., & Milgrom, P. R. (1985). *Bid, Ask and Transaction Prices in a Specialist Market with Heterogeneously Informed Traders.* Journal of Financial Economics, 14(1), 71–100.
3. Natenberg, S. (1994). *Option Volatility and Pricing.* McGraw-Hill.
4. Gatheral, J. (2006). *The Volatility Surface: A Practitioner's Guide.* Wiley Finance.
5. NSE India. *BANKNIFTY Option Chain Data.* https://www.nseindia.com/option-chain

---

## Author

**Shriraj Aher**

*Quant Research Project — Indian Derivatives Markets*

---

> *"Implied volatility is not implied by the market. It is implied by the model you use to back it out — and the data you feed into it."*
