# HRP + Spectral Clustering Portfolio Optimizer

EEIF, Emory University.

Takes a list of holdings and proposes target weights using Hierarchical Risk Parity (HRP). Stocks are first grouped into clusters by a spectral clustering model trained on a wider universe of ~255 stocks. A static dashboard shows the results.

## Quick start

```bash
pip install numpy pandas scipy scikit-learn
python main.py                      # writes portfolio_results.json
python -m http.server 8000          # then open http://localhost:8000/portfolio_dashboard.html
```

The dashboard loads `portfolio_results.json` with `fetch()`. Opening the HTML file directly (`file://`) is blocked by most browsers, so serve the folder as shown above.

## Files

| File | Purpose |
|---|---|
| `main.py` | The optimizer. Reads prices, clusters, allocates, writes JSON. |
| `training_data.csv` | Daily adjusted close prices (2021-07-29 to 2026-02-20), one column per ticker. |
| `portfolio_results.json` | Output of `main.py`. Input to the dashboard. |
| `portfolio_dashboard.html` | Interactive dashboard (reads the JSON). |
| `portfolio_optimization.png` | Static chart. |
| `model_documentation (1).pdf` | Full methodology write-up. |

## How it works

1. **Load prices.** Forward-fill gaps, drop weekend rows, compute log returns.
2. **Remove the market factor.** Regress each stock on the equal-weighted average return of the training universe and keep the residuals. This stops the market-wide correlation from swamping industry and regional structure.
3. **Spectral clustering.** Convert residual correlations to distances, `d = sqrt(0.5 * (1 - corr))`. Build a Gaussian-similarity kNN graph (k=3) and take the normalized Laplacian. Sweep the kernel width sigma and pick the cluster count that is most stable across the sweep (eigengap). Run k-means on the eigenvectors. Any cluster over 40% of the universe is re-clustered.
4. **Map the portfolio.** Holdings in the training set take their cluster directly. Others are projected onto the cluster with the lowest average correlation distance.
5. **HRP allocation.** Order holdings by Ward linkage on correlation distance. Recursively bisect, splitting weight between halves in inverse proportion to their variance.
6. **Cap and compare.** Cap any position at `MAX_WEIGHT` (default 15%) and redistribute the excess. Compare risk (volatility, HHI, effective N, diversification ratio, risk contribution by cluster) for current, HRP and equal-weight portfolios. Trades are the dollar difference between HRP target and current value at the last price.

## Technical details

Notation: `r_i,t` is the daily log return of stock `i`, `N` the number of training stocks, `n` the number of portfolio holdings.

### Returns and market-factor removal

```
r_i,t = ln(P_i,t / P_i,t-1)
m_t   = (1/N) * sum_i r_i,t                    equal-weighted market return
beta_i  = Cov(r_i, m) / Var(m)
alpha_i = mean(r_i) - beta_i * mean(m)
e_i,t   = r_i,t - (alpha_i + beta_i * m_t)     residual used for clustering
```

The reported `variance_explained` is `1 - mean(Var(e_i)) / mean(Var(r_i))`.

### Distance

```
rho_ij = Corr(e_i, e_j)
d_ij   = sqrt(0.5 * (1 - rho_ij))
```

`d` is 0 for perfectly correlated stocks, about 0.71 for uncorrelated, and 1 for perfectly anti-correlated. It satisfies the triangle inequality, so it is a proper metric.

### Spectral clustering

```
S_ij = exp(-d_ij^2 / (2 * sigma^2)),  S_ii = 0       Gaussian similarity
W_ij = S_ij if j is among i's 3 nearest neighbours   (symmetrised: W_ij = W_ji)
D    = diag(sum_j W_ij)                              degree matrix
L_rw = I - D^-1 W                                    random-walk Laplacian
```

Eigenvalues `0 = lambda_1 <= lambda_2 <= ...` of `L_rw` are sorted ascending. A large gap after the first `k` eigenvalues means the graph has about `k` loosely connected groups.

**Choosing `k` and `sigma`.** `sigma` is swept over 60 evenly spaced values between the 5th and 95th percentile of all pairwise distances. At each `sigma`, take the largest gap in `diff(lambda_1..lambda_20)`, skipping the first gap, and call its position `k(sigma)`. The final `k` is the most frequent value across the sweep, and the final `sigma` is the median of the sigmas that voted for it.

**Assigning clusters.** Stack eigenvectors 2 to `k+1` as an `N x k` embedding and run k-means (`random_state=42`, `n_init=30`).

**Splitting oversized clusters.** If a cluster holds more than 40% of the universe (and more than 15 stocks), repeat the whole procedure on its sub-distance matrix with `knn = max(3, round(ln(size)))`. Repeat until no cluster is oversized.

### Projecting holdings not in the training set

For a holding `p` with residual series `e_p`, compute `d_pj` against every training stock `j`, then assign `p` to the cluster `c` with the smallest mean distance:

```
cluster(p) = argmin_c  mean_{j in c} d_pj
```

### HRP allocation

1. Build the holdings' correlation distance matrix `d = sqrt(0.5 * (1 - rho))` from raw (not residual) returns, then Ward linkage. The leaf order of the dendrogram gives a quasi-diagonal ordering, so similar holdings sit next to each other.
2. Recursively split the ordered list in half, `L` and `R`. Each half's variance comes from its own internal allocation `w`:

```
V(S) = w_S' * Sigma_S * w_S
alpha = 1 - V(L) / (V(L) + V(R))
w_i  = alpha * w_i^L        for i in L
w_i  = (1 - alpha) * w_i^R  for i in R
```

The lower-variance half gets the larger share. A single stock gets weight 1. `Sigma` is the sample covariance of daily log returns.

3. **Weight cap.** Up to 100 iterations: `excess = sum(max(w_i - cap, 0))`, clip weights to `cap`, redistribute `excess` to uncapped names in proportion to their current weights, then renormalise to sum to 1.

### Risk metrics

Annualised with 252 trading days: `Sigma_a = 252 * Sigma`, `sigma_i = sqrt(252) * std(r_i)`.

```
portfolio vol          sigma_p = sqrt(w' * Sigma_a * w)
HHI                    sum_i w_i^2
effective N            1 / HHI
diversification ratio  (sum_i w_i * sigma_i) / sigma_p
risk contribution      RC_i = w_i * (Sigma_a * w)_i / sigma_p^2     (sums to 1)
cluster risk           sum of RC_i over holdings in the cluster
```

### Trades

```
current_value_i = shares_i * last_price_i
target_value_i  = w_hrp_i * sum(current_value)
trade_i         = target_value_i - current_value_i      (negative = sell)
```

## Configuring

Edit the top of `main.py`:

- `PORTFOLIO`: `'TICKER': ('Display Name', shares)`. The ticker must be a column in `training_data.csv`.
- `MAX_WEIGHT`: single-position cap.
- `CSV_PATH`: price file location.
- `TICKER_META`: `(market, sector)` per ticker, used only to label clusters. Add entries for new tickers.

To add a stock, add a price column for it to `training_data.csv` (dates as the index, matching the existing format) and then add it to `PORTFOLIO` and `TICKER_META`.

The code below the "model engine" banner in `main.py` is marked as do-not-touch. The clustering and allocation steps are tuned together, so change it deliberately and compare outputs.

## Output

`portfolio_results.json` contains, among other things: per-position current and HRP weights and suggested trades, cluster membership and labels, risk metrics for the three portfolios, per-cluster risk contributions, the eigenvalue spectrum, the sigma sweep, and the correlation matrix.

## Caveats

- Weights are based on historical covariance and are not a forecast. HRP does not use expected returns.
- Trade amounts ignore taxes, transaction costs, lot sizes and fractional-share limits.
- Prices are static as of the last row of `training_data.csv`. Refresh the CSV before relying on the output.
- Cluster labels come from the two most common sectors in `TICKER_META`, so they are only as good as that mapping.
- Not investment advice.
