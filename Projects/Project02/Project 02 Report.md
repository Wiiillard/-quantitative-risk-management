# Project 02 Report

***Name: Kefan Liang NetID: kl508***



## Problem 1: Calculating Arithmetic and Log Returns

### Introduction
In this problem, we are given a dataset (`DailyPrices.csv`) containing daily stock prices for several stocks, including `SPY`, `AAPL`, and `EQIX`. The tasks are to calculate the arithmetic and log returns for these stocks, remove the mean from each series, and present the last 5 rows along with the total standard deviation.

---

### Task 1A: Calculate Arithmetic Returns and Remove Mean

#### Steps:
1. **Load the dataset**: We start by loading the dataset using `pandas`.
2. **Extract relevant stocks**: We extract the stock prices for `SPY`, `AAPL`, and `EQIX`.
3. **Calculate arithmetic returns**: Arithmetic returns are calculated using the `pct_change()` method.
4. **Remove the mean**: We subtract the mean from each return series to ensure each series has a mean of 0.
5. **Present the last 5 rows and standard deviation**: We display the last 5 rows of the mean-adjusted arithmetic returns and calculate the standard deviation.

#### Results:
- **Last 5 rows of mean-adjusted arithmetic returns**:
  ```
          SPY      AAPL      EQIX
  499 -0.011492 -0.014678 -0.006966
  500 -0.012377 -0.014699 -0.008064
  501 -0.004603 -0.008493  0.006512
  502 -0.003422 -0.027671  0.000497
  503  0.011538 -0.003445  0.015745
  ```

- **Standard deviation of arithmetic returns**:
  ```
  SPY     0.008077
  AAPL    0.013483
  EQIX    0.015361
  dtype: float64
  ```

---

### Task 1B: Calculate Log Returns and Remove Mean

#### Steps:
1. **Calculate log returns**: Log returns are calculated using the formula `log(P_t / P_{t-1})`.
2. **Remove the mean**: We subtract the mean from each log return series to ensure each series has a mean of 0.
3. **Present the last 5 rows and standard deviation**: We display the last 5 rows of the mean-adjusted log returns and calculate the standard deviation.

#### Results:
- **Last 5 rows of mean-adjusted log returns**:
  ```
          SPY      AAPL      EQIX
  499 -0.011515 -0.014675 -0.006867
  500 -0.012410 -0.014696 -0.007972
  501 -0.004577 -0.008427  0.006602
  502 -0.003392 -0.027930  0.000613
  503  0.011494 -0.003356  0.015725
  ```

- **Standard deviation of log returns**:
  ```
  SPY     0.008078
  AAPL    0.013446
  EQIX    0.015270
  dtype: float64
  ```

### 
In this problem, we successfully calculated the arithmetic and log returns for the stocks `SPY`, `AAPL`, and `EQIX`. After removing the mean from each return series, we presented the last 5 rows of the mean-adjusted returns and calculated the standard deviation for each series. The results show that the standard deviations of the arithmetic and log returns are very similar, which is expected for small returns.

# 

## Problem 2: Portfolio Valuation and Risk Analysis

### Introduction
In this problem, we are given a portfolio consisting of:
- 100 shares of SPY
- 200 shares of AAPL
- 150 shares of EQIX

The tasks are:
1. Calculate the current value of the portfolio as of January 3, 2025.
2. Calculate the Value at Risk (VaR) and Expected Shortfall (ES) for each stock and the entire portfolio at the 5% alpha level using three different methods:
   - Normally distributed returns with exponentially weighted covariance (lambda = 0.97)
   - T-distribution using a Gaussian Copula
   - Historical simulation using the full history
3. Discuss the differences between the methods.

---

### Task 2A: Calculate the Current Portfolio Value

#### Steps:
1. **Load the dataset**: We load the `DailyPrices.csv` dataset and extract the prices for the relevant stocks (`SPY`, `AAPL`, and `EQIX`).
2. **Calculate the portfolio value**: We compute the total value of the portfolio as of January 3, 2025, using the given holdings.

#### Results:
- **Portfolio Value on 2025-01-03**: $251,862.50

---

### Task 2B: Calculate VaR and ES Using Different Methods

#### Method 1: Normal Distribution with Exponentially Weighted Covariance (Lambda = 0.97)

##### Steps:
1. **Compute exponentially weighted covariance matrix**: We calculate the covariance matrix using an exponential weighting factor (lambda = 0.97).
2. **Calculate VaR and ES**: Using the normal distribution assumption, we compute the VaR and ES for each stock and the portfolio.

##### Results:
- **SPY VaR**: $786.55
- **SPY ES**: $-986.37
- **AAPL VaR**: $1,076.50
- **AAPL ES**: $-1,349.97
- **EQIX VaR**: $3,616.78
- **EQIX ES**: $-4,535.59
- **Portfolio VaR**: $3,892.41
- **Portfolio ES**: $-4,881.24

---

#### Method 2: T-Distribution Using a Gaussian Copula

##### Steps:
1. **Fit t-distribution to each stock's returns**: We fit a t-distribution to the returns of each stock.
2. **Convert returns to uniform scale**: Using the fitted t-distributions, we transform the returns to a uniform scale.
3. **Generate multivariate normal samples**: We generate samples from a multivariate normal distribution using the correlation matrix.
4. **Convert back to t-distribution returns**: We transform the uniform samples back to t-distribution returns.
5. **Calculate VaR and ES**: Using the simulated returns, we compute the VaR and ES for each stock and the portfolio.

##### Results:
- **SPY VaR**: $721.26
- **SPY ES**: $1,008.64
- **AAPL VaR**: $962.98
- **AAPL ES**: $1,388.15
- **EQIX VaR**: $3,389.79
- **EQIX ES**: $4,957.79
- **Portfolio VaR**: $4,197.49
- **Portfolio ES**: $6,114.05

---

#### Method 3: Historical Simulation

##### Steps:
1. **Calculate VaR and ES using historical returns**: We use the full history of returns to compute the VaR and ES for each stock and the portfolio.

##### Results:
- **SPY VaR**: $820.93
- **SPY ES**: $1,032.46
- **AAPL VaR**: $1,007.60
- **AAPL ES**: $1,388.98
- **EQIX VaR**: $3,545.28
- **EQIX ES**: $4,660.30
- **Portfolio VaR**: $4,364.04
- **Portfolio ES**: $5,887.27

---

### Task 2C: Discussion of Differences Between Methods

#### Normal Distribution Method:
- **Assumption**: Returns are normally distributed.
- **Pros**: Simple and computationally efficient.
- **Cons**: May underestimate risk if returns have fat tails or are not normally distributed.
- **Results**: The VaR and ES values are generally lower compared to the other methods, indicating less extreme risk.

#### T-Distribution with Gaussian Copula:
- **Assumption**: Returns follow a t-distribution, and dependencies are modeled using a Gaussian Copula.
- **Pros**: Captures fat tails in returns and models dependencies more accurately.
- **Cons**: More computationally intensive than the normal distribution method.
- **Results**: The VaR and ES values are higher than the normal distribution method, reflecting the higher risk due to fat tails.

#### Historical Simulation:
- **Assumption**: Historical returns are a good representation of future returns.
- **Pros**: No assumptions about the distribution of returns; captures all historical patterns.
- **Cons**: Relies heavily on historical data, which may not always predict future risks accurately.
- **Results**: The VaR and ES values are intermediate between the normal and t-distribution methods, reflecting a balance between the two approaches.

### 
In this problem, we calculated the current value of the portfolio and estimated the VaR and ES using three different methods. The normal distribution method provided the most conservative estimates, while the t-distribution with Gaussian Copula captured more extreme risks due to fat tails. The historical simulation method provided a middle ground, relying on actual historical data. Each method has its strengths and limitations, and the choice of method depends on the specific requirements and assumptions of the analysis.

---



## Problem 3: Option Pricing and Risk Analysis

### Introduction
In this problem, we analyze a European Call option with the following parameters:
- Time to maturity: 3 months (0.25 years)
- Call Price: $3.00
- Stock Price: $31
- Strike Price: $30
- Risk-Free Rate: 10%
- No dividends are paid.

The tasks are:
1. Calculate the implied volatility.
2. Calculate the Delta, Vega, and Theta of the option. Determine the approximate change in the option price if the implied volatility increases by 1%.
3. Calculate the price of the put option using the Generalized Black-Scholes-Merton formula and verify if Put-Call Parity holds.
4. Calculate the Value at Risk (VaR) and Expected Shortfall (ES) for a portfolio consisting of 1 call, 1 put, and 1 share of stock over a 20-trading-day holding period at a 5% alpha level using:
   - Delta-Normal Approximation
   - Monte Carlo Simulation
5. Discuss the differences between the two methods.

---

### Task 3A: Calculate the Implied Volatility

#### Steps:
1. **Use the Black-Scholes formula** to calculate the implied volatility of the call option.
2. **Newton-Raphson method** is used to iteratively find the implied volatility that matches the market price of the call option.

#### Results:
- **Implied Volatility**: 0.335 (33.5%)

---

### Task 3B: Calculate Delta, Vega, and Theta

#### Steps:
1. **Delta**: Measures the sensitivity of the option price to changes in the stock price.
2. **Vega**: Measures the sensitivity of the option price to changes in volatility.
3. **Theta**: Measures the sensitivity of the option price to the passage of time.

#### Results:
- **Delta**: 0.666
- **Vega**: 5.641
- **Theta**: -5.545

#### Price Change Due to 1% Volatility Increase:
- **Price Change**: $0.056 (calculated using Vega)

---

### Task 3C: Calculate the Put Price and Verify Put-Call Parity

#### Steps:
1. **Calculate the put price** using the Generalized Black-Scholes-Merton formula.
2. **Verify Put-Call Parity**: Check if the relationship \( $C + K e^{-rT} = P + S$ \) holds, where \( $C$ \) is the call price, \( $P$ \) is the put price, \( $S$ \) is the stock price, and \( K \) is the strike price.

#### Results:
- **Put Price**: $1.259
- **Put-Call Parity Difference**: \( $1.76 \times 10^{-10}$ \) (essentially zero, confirming Put-Call Parity holds)

---

### Task 3D: Calculate VaR and ES

#### Method 1: Delta-Normal Approximation

##### Steps:
1. **Calculate daily volatility** using the implied volatility.
2. **Compute VaR and ES** using the normal distribution assumption.

##### Results:
- **VaR (Delta-Normal)**: -$4.785
- **ES (Delta-Normal)**: -$6.001

---

#### Method 2: Monte Carlo Simulation

##### Steps:
1. **Simulate stock price paths** using the implied volatility.
2. **Calculate portfolio losses** for each simulated path.
3. **Compute VaR and ES** from the simulated losses.

##### Results:
- **VaR (Monte Carlo)**: -$5.183
- **ES (Monte Carlo)**: -$6.625

---

### Task 3E: Discussion of Differences Between Methods

#### Delta-Normal Approximation:
- **Assumption**: Returns are normally distributed.
- **Pros**: Simple and computationally efficient.
- **Cons**: May underestimate risk if returns have fat tails or are not normally distributed.
- **Results**: The VaR and ES values are slightly less extreme compared to the Monte Carlo method.

#### Monte Carlo Simulation:
- **Assumption**: Simulates a large number of possible future stock price paths.
- **Pros**: Captures non-linearities and fat tails in returns.
- **Cons**: Computationally intensive.
- **Results**: The VaR and ES values are slightly higher, reflecting a more accurate representation of risk.

### 
In this problem, we calculated the implied volatility, Delta, Vega, and Theta of a European Call option. We also calculated the price of the corresponding put option and verified that Put-Call Parity holds. Using both Delta-Normal Approximation and Monte Carlo Simulation, we estimated the VaR and ES for a portfolio consisting of 1 call, 1 put, and 1 share of stock. The Delta-Normal method provided a simpler but slightly less accurate estimate, while the Monte Carlo method captured more extreme risks due to its ability to model non-linearities and fat tails.

---

