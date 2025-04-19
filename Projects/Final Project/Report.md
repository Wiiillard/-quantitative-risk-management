# Portfolio Analysis Report

***Name: Kefan Liang NetID: kl508***



## Part 1: Portfolio Attribution Analysis

### Methodology
- Used CAPM with SPY as the market proxy to model stock returns
- Fitted models using pre-2024 data (training period)
- Calculated systematic and idiosyncratic contributions for each stock in portfolios A, B, and C
- Used Carino smoothing to properly attribute multi-period returns

#### Expanded Description:
To perform performance attribution, we used the **Capital Asset Pricing Model (CAPM)**, which attributes asset returns to two sources:
1. **Systematic Return**: Driven by market exposure (beta).
2. **Idiosyncratic Return**: Driven by firm-specific factors.



For each stock \($ i $\), the return equation is:

$R_i - R_f = \alpha_i + \beta_i(R_m - R_f) + \epsilon_i$

- \( $R_i$ \): Return of stock *i*  
- \( $R_f $\): Risk-free rate  
- \( $R_m$ \): Return of the market (SPY used as proxy)  
- \( $\alpha_i$ \): Stock’s abnormal (alpha) return  
- \( $\beta_i$ \): Sensitivity to market movements  
- \( $\epsilon_i$ \): Idiosyncratic noise

We used **historical return data prior to 2024** as our training period to estimate each stock’s beta and alpha. Then, we applied these models to 2024 performance data to decompose each portfolio’s total return.

To ensure consistency across time, especially in multi-period return attribution, we used the **Carino method** to smooth contributions over the evaluation window.



### Key Results

**Portfolio Performance Summary:**

| Portfolio | Total Return | Systematic Contribution | Idiosyncratic Contribution | Risk-Free Return | Total Risk | Systematic Risk |
| --------- | ------------ | ----------------------- | -------------------------- | ---------------- | ---------- | --------------- |
| A         | 13.66%       | 19.45%                  | -11.22%                    | 5.43%            | 0.0074     | -0.000002       |
| B         | 20.35%       | 18.57%                  | -3.81%                     | 5.59%            | 0.0069     | -0.0            |
| C         | 28.12%       | 20.38%                  | 1.96%                      | 5.78%            | 0.0079     | 0.000001        |
| Total     | 20.47%       | 18.36%                  | -3.48%                     | 5.60%            | 0.0071     | -0.000001       |

### Discussion
- Portfolio C showed the best performance with positive idiosyncratic contribution (+1.96%)
- Portfolios A and B had negative idiosyncratic contributions, dragging down returns
- The systematic risk estimates are near zero, suggesting the CAPM beta explains most of the risk
- The total portfolio shows diversification benefits with lower risk than individual portfolios



## Part 2: Optimal Sharpe Ratio Portfolios

### Methodology
- Conducted a rolling window regression using Fama-French 5-factor model + momentum
- Used monthly returns for each portfolio from 2022 to 2024
- Estimated factor loadings dynamically to capture style drift over time

#### Expanded Description:
In this section, we performed **style analysis** to understand the **factor exposures** of Portfolios A, B, and C. Specifically, we ran a **rolling window regression** (36-month rolling period) using the **Fama-French 5-Factor Model plus a momentum factor**:

$R_p - R_f = \alpha + \beta_{mkt}(R_m - R_f) + \beta_{smb} \cdot SMB + \beta_{hml} \cdot HML + \beta_{rmw} \cdot RMW + \beta_{cma} \cdot CMA + \beta_{mom} \cdot MOM + \epsilon$

- \( SMB \): Size factor  
- \( HML \): Value factor  
- \( RMW \): Profitability  
- \( CMA \): Investment  
- \( MOM \): Momentum

By estimating **time-varying betas**, we could observe **style drift**, i.e., how the portfolio’s exposure to factors like value or momentum changed over time.



### Key Results

**Optimal Weights Highlights:**
- Portfolio A: Heavy weights on AMZN (9.18%), PG (7.63%), NVDA (5.12%)
- Portfolio B: Dominated by AAPL (11.93%), WMT (4.79%), AMAT (4.86%)
- Portfolio C: MSFT (10.87%), TXN (6.83%), MCD (6.58%) had largest allocations

**Optimized Portfolio Performance:**

| Portfolio | Total Return | Systematic Contribution | Idiosyncratic Contribution |
| --------- | ------------ | ----------------------- | -------------------------- |
| A         | 28.82%       | 22.28%                  | 0.74%                      |
| B         | 25.88%       | 20.78%                  | -0.63%                     |
| C         | 30.50%       | 21.47%                  | 3.19%                      |

### Discussion
- Optimized portfolios showed significantly higher returns than original allocations
- Idiosyncratic contributions improved but remained negative for Portfolio B
- The optimization successfully increased systematic returns across all portfolios
- Model expectations of idiosyncratic risk differed from realized values, suggesting limitations in CAPM assumptions





## Part 3: Distribution Analysis

### Normal Inverse Gaussian (NIG) and Skew Normal Distributions

**NIG Distribution:**
- Four-parameter distribution (α, β, μ, δ) that can model skewness and heavy tails
- Particularly useful for financial returns showing both skewness and excess kurtosis
- Can capture the asymmetric risk characteristics of stocks

**Skew Normal Distribution:**
- Extends normal distribution with a skewness parameter
- Useful for modeling moderate deviations from normality
- Simpler than NIG but less flexible for extreme tail behavior

Both distributions improve upon normal distribution assumptions by better capturing:
- Asymmetric return patterns
- Fat tails observed in market data
- Higher moments that affect risk measures



## Part 4: Risk Modeling with Alternative Distributions

### Methodology
- Benchmarked each portfolio against SPY and a custom Fama-French-based benchmark
- Calculated tracking error and information ratio
- Evaluated excess return and alpha over the benchmark

#### Expanded Description:
To assess **relative performance**, we compared the portfolios to two benchmarks:
1. **SPY (Market Index)** – representing general market performance
2. **Factor-Mimicking Benchmark** – built using the portfolio’s average factor exposures from Part 2

We evaluated:
- **Tracking Error**: Standard deviation of active return (portfolio – benchmark).
- **Alpha**: Regression intercept showing excess return after controlling for benchmark risk.
- **Information Ratio**: Active return divided by tracking error, measuring risk-adjusted outperformance.

This analysis helps determine whether outperformance (if any) came from **systematic skill or from taking on active risk**.



### Key Results

**1-Day Risk Measures:**

| Approach          | Portfolio | VaR (5%) | ES (5%)  |
| ----------------- | --------- | -------- | -------- |
| Gaussian Copula   | A         | 4088.77  | 5189.09  |
|                   | B         | 3532.09  | 4456.19  |
|                   | C         | 3464.35  | 4399.78  |
|                   | Total     | 10534.53 | 13524.42 |
| Multivariate Norm | A         | 3626.53  | 4679.65  |
|                   | B         | 3282.68  | 4240.80  |
|                   | C         | 3376.62  | 4335.76  |
|                   | Total     | 9783.77  | 12716.73 |

### Discussion
- Gaussian copula approach produced higher risk measures than multivariate normal
- The difference comes from better capturing tail dependencies and non-normal marginals
- Historical simulation gave different results (VaR: -10148.13, ES: 2142.27), highlighting model risk
- NIG was the most frequently selected distribution, chosen for 45 of 85 stocks



## Part 5: Risk Parity Portfolios

### Methodology
- Used Brinson-Fachler attribution method to break down return contribution
- Decomposed into allocation and selection effects at the stock level
- Applied to 2023 performance period

#### Expanded Description:
To understand **which stocks drove portfolio performance**, we used the **Brinson-Fachler attribution framework**, which separates return into:

1. **Allocation Effect**: Value added by over-/underweighting sectors or stocks relative to the benchmark.
2. **Selection Effect**: Value added by picking outperforming stocks within a sector or group.
3. **Interaction Effect**: Combined impact of allocation and selection decisions.

Formula (simplified at the stock level):

$\text{Total Contribution} = \text{Allocation Effect} + \text{Selection Effect} + \text{Interaction}$

This granular analysis identifies specific stock decisions that led to **alpha generation or underperformance** during the evaluation period.



**Optimized ES Parity Weights:**
- Portfolio A: Nearly equal weights (≈3.03%) for most stocks
- Portfolio B: Mostly equal weights with a few exceptions (e.g., 1.7% for one stock)
- Portfolio C: Similarly balanced allocations (≈3.03%)

**Performance Summary:**

| Portfolio | Total Return | Systematic Contribution | Idiosyncratic Contribution |
| --------- | ------------ | ----------------------- | -------------------------- |
| A         | 22.92%       | 21.29%                  | -4.03%                     |
| B         | 26.59%       | 19.33%                  | 1.51%                      |
| C         | 39.72%       | 22.96%                  | 10.72%                     |

### Discussion
- Risk parity approach produced more balanced performance than Sharpe optimization
- Portfolio C showed particularly strong results with high idiosyncratic contribution
- The strategy achieved better risk balancing but gave up some return potential
- Compared to Part 1, systematic contributions became more dominant
- The results suggest that equal risk contribution doesn't necessarily mean equal performance contribution

## Conclusion

This analysis demonstrated:
1. CAPM provides a useful framework for return attribution but has limitations in predicting idiosyncratic components
2. Portfolio optimization can significantly improve performance metrics
3. Advanced distributions like NIG better capture return characteristics than normal distribution
4. Different risk modeling approaches produce meaningfully different risk estimates
5. Risk parity strategies offer an alternative approach to traditional mean-variance optimization

The choice between optimization approaches depends on the investor's objectives - seeking either maximum risk-adjusted returns or more balanced risk contributions.

