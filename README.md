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
