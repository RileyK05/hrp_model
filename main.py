"""
HRP + Spectral Clustering Portfolio Optimizer
EEIF -- Emory University


Outputs:
  - portfolio_results.json  (data for dashboard)
  - portfolio_dashboard.png (static backup chart)
"""

import numpy as np
import pandas as pd
import json
from scipy.linalg import eigh
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.cluster import KMeans
from collections import Counter
import warnings, sys, os
warnings.filterwarnings('ignore')


# =============================================================================
#  PORTFOLIO HOLDINGS  --  OK TO EDIT
# =============================================================================
# Format: 'TICKER': ('Display Name', number_of_shares)
# Add/remove positions here when the portfolio changes.

PORTFOLIO = {
    'GOOG':  ('Alphabet',            42),
    'AMZN':  ('Amazon',              22),
    'NVDA':  ('Nvidia',              18),
    'META':  ('Meta Platforms',       5),
    'MSFT':  ('Microsoft',            7),
    'TSM':   ('Taiwan Semi',         10),
    'ZETA':  ('Zeta Global',        275),
    'SHOP':  ('Shopify',             20),
    'UBER':  ('Uber',                41),
    'HOOD':  ('Robinhood',           50),
    'COIN':  ('Coinbase',             5),
    'JPM':   ('JP Morgan',           21),
    'BRK-B': ('Berkshire Hathaway',   7),
    'MKL':   ('Markel',               1),
    'IBKR':  ('Interactive Brokers', 25),
    'EFX':   ('Equifax',             10),
    'JNJ':   ('Johnson & Johnson',   18),
    'UNH':   ('United Health',        5),
    'HDB':   ('HDFC Bank',           54),
    'IBN':   ('ICICI Bank',          59),
    'KWEB':  ('China Internet ETF',  91),
    'NE':    ('Noble Corp',          19),
    'HCC':   ('Warrior Met Coal',     8),
    'AMR':   ('Alpha Metallurgical',  3),
    'VAL':   ('Valaris',             10),
    'TDW':   ('TideWater',           9),
    'BN':    ('Brookfield Corp',    154),
    'SHV':   ('Blackrock T-Bill',    60),
}


# =============================================================================
#  SETTINGS  --  OK TO EDIT
# =============================================================================

CSV_PATH   = 'training_data.csv'   # price data file
MAX_WEIGHT = 0.15                  # max 15% in any single position


# =============================================================================
#  TICKER METADATA  --  OK TO EDIT (add new tickers as needed)
# =============================================================================
# Maps each training ticker to (market, sector) for cluster labeling.

TICKER_META = {
    'AAPL':('US','Mega Cap Tech'),'MSFT':('US','Mega Cap Tech'),
    'GOOG':('US','Mega Cap Tech'),'AMZN':('US','Mega Cap Tech'),
    'META':('US','Mega Cap Tech'),'NVDA':('US','Mega Cap Tech'),
    'AVGO':('US','Mega Cap Tech'),'ORCL':('US','Mega Cap Tech'),
    'CRM':('US','Mega Cap Tech'),'ADBE':('US','Mega Cap Tech'),
    'AMD':('US','Mega Cap Tech'),'INTC':('US','Mega Cap Tech'),
    'CSCO':('US','Mega Cap Tech'),'TXN':('US','Mega Cap Tech'),
    'QCOM':('US','Mega Cap Tech'),
    'NOW':('US','SaaS / Growth'),'PLTR':('US','SaaS / Growth'),
    'PANW':('US','SaaS / Growth'),'CRWD':('US','SaaS / Growth'),
    'DDOG':('US','SaaS / Growth'),'NET':('US','SaaS / Growth'),
    'ZS':('US','SaaS / Growth'),'SHOP':('US','SaaS / Growth'),
    'PYPL':('US','SaaS / Growth'),'SNOW':('US','SaaS / Growth'),
    'HUBS':('US','SaaS / Growth'),'VEEV':('US','SaaS / Growth'),
    'WDAY':('US','SaaS / Growth'),
    'COIN':('US','Fintech / Crypto'),'HOOD':('US','Fintech / Crypto'),
    'SOFI':('US','Fintech / Crypto'),'AFRM':('US','Fintech / Crypto'),
    'MARA':('US','Fintech / Crypto'),'RIOT':('US','Fintech / Crypto'),
    'JPM':('US','Trad Financials'),'BAC':('US','Trad Financials'),
    'WFC':('US','Trad Financials'),'GS':('US','Trad Financials'),
    'MS':('US','Trad Financials'),'C':('US','Trad Financials'),
    'BLK':('US','Trad Financials'),'SCHW':('US','Trad Financials'),
    'AXP':('US','Trad Financials'),'COF':('US','Trad Financials'),
    'USB':('US','Trad Financials'),'PNC':('US','Trad Financials'),
    'BRK-B':('US','Trad Financials'),'MKL':('US','Trad Financials'),
    'AIG':('US','Trad Financials'),'ALL':('US','Trad Financials'),
    'PRU':('US','Trad Financials'),'MET':('US','Trad Financials'),
    'CB':('US','Trad Financials'),'IBKR':('US','Trad Financials'),
    'EFX':('US','Trad Financials'),'MCO':('US','Trad Financials'),
    'JNJ':('US','Healthcare'),'UNH':('US','Healthcare'),
    'PFE':('US','Healthcare'),'MRK':('US','Healthcare'),
    'ABBV':('US','Healthcare'),'LLY':('US','Healthcare'),
    'TMO':('US','Healthcare'),'ABT':('US','Healthcare'),
    'DHR':('US','Healthcare'),'BMY':('US','Healthcare'),
    'AMGN':('US','Healthcare'),'GILD':('US','Healthcare'),
    'ISRG':('US','Healthcare'),'REGN':('US','Healthcare'),
    'VRTX':('US','Healthcare'),'ZTS':('US','Healthcare'),
    'MDT':('US','Healthcare'),'SYK':('US','Healthcare'),
    'EW':('US','Healthcare'),'HCA':('US','Healthcare'),
    'MRNA':('US','Biotech'),'BIIB':('US','Biotech'),
    'ALNY':('US','Biotech'),'IONS':('US','Biotech'),
    'PCVX':('US','Biotech'),'NBIX':('US','Biotech'),
    'SRPT':('US','Biotech'),'BMRN':('US','Biotech'),
    'INCY':('US','Biotech'),'ARGX':('US','Biotech'),
    'SMMT':('US','Biotech'),'UTHR':('US','Biotech'),
    'HALO':('US','Biotech'),'LEGN':('US','Biotech'),
    'BPMC':('US','Biotech'),'EXAS':('US','Biotech'),
    'RARE':('US','Biotech'),
    'WMT':('US','Consumer'),'PG':('US','Consumer'),
    'KO':('US','Consumer'),'PEP':('US','Consumer'),
    'COST':('US','Consumer'),'HD':('US','Consumer'),
    'MCD':('US','Consumer'),'NKE':('US','Consumer'),
    'SBUX':('US','Consumer'),'TGT':('US','Consumer'),
    'LOW':('US','Consumer'),'CL':('US','Consumer'),
    'DG':('US','Consumer'),'DLTR':('US','Consumer'),
    'XOM':('US','Energy'),'CVX':('US','Energy'),
    'COP':('US','Energy'),'SLB':('US','Energy'),
    'EOG':('US','Energy'),'MPC':('US','Energy'),
    'VLO':('US','Energy'),'PSX':('US','Energy'),
    'HAL':('US','Energy'),'OXY':('US','Energy'),
    'DVN':('US','Energy'),'FANG':('US','Energy'),
    'NE':('US','Energy Svcs'),'VAL':('US','Energy Svcs'),
    'TDW':('US','Energy Svcs'),'HCC':('US','Energy Svcs'),
    'AMR':('US','Energy Svcs'),'BTU':('US','Energy Svcs'),
    'CAT':('US','Industrials'),'DE':('US','Industrials'),
    'HON':('US','Industrials'),'UNP':('US','Industrials'),
    'BA':('US','Industrials'),'GE':('US','Industrials'),
    'EMR':('US','Industrials'),'ETN':('US','Industrials'),
    'WM':('US','Industrials'),'RSG':('US','Industrials'),
    'URI':('US','Industrials'),'FAST':('US','Industrials'),
    'ROK':('US','Industrials'),'DOV':('US','Industrials'),
    'AME':('US','Industrials'),'ITW':('US','Industrials'),
    'PH':('US','Industrials'),'CARR':('US','Industrials'),
    'OTIS':('US','Industrials'),'TT':('US','Industrials'),
    'PCAR':('US','Industrials'),'CMI':('US','Industrials'),
    'GNRC':('US','Industrials'),'WAB':('US','Industrials'),
    'FTV':('US','Industrials'),'NDSN':('US','Industrials'),
    'SWK':('US','Industrials'),'GWW':('US','Industrials'),
    'AOS':('US','Industrials'),'XYL':('US','Industrials'),
    'RTX':('US','Aerospace/Def'),'LMT':('US','Aerospace/Def'),
    'GD':('US','Aerospace/Def'),'NOC':('US','Aerospace/Def'),
    'HII':('US','Aerospace/Def'),'TDG':('US','Aerospace/Def'),
    'HWM':('US','Aerospace/Def'),'AXON':('US','Aerospace/Def'),
    'HEI':('US','Aerospace/Def'),'LDOS':('US','Aerospace/Def'),
    'SPG':('US','REITs'),'PLD':('US','REITs'),
    'AMT':('US','REITs'),'O':('US','REITs'),
    'EQIX':('US','REITs'),'DLR':('US','REITs'),
    'PSA':('US','REITs'),'BN':('US','REITs/Infra'),
    'NEE':('US','Utilities'),'DUK':('US','Utilities'),
    'SO':('US','Utilities'),'D':('US','Utilities'),
    'AEP':('US','Utilities'),'XEL':('US','Utilities'),
    'SRE':('US','Utilities'),
    'ZETA':('US','SmallMid Growth'),'UBER':('US','SmallMid Growth'),
    'ABNB':('US','SmallMid Growth'),'RBLX':('US','SmallMid Growth'),
    'PATH':('US','SmallMid Growth'),'BILL':('US','SmallMid Growth'),
    'GLBE':('US','SmallMid Growth'),
    'BABA':('China','Internet/Tech'),'JD':('China','Internet/Tech'),
    'PDD':('China','Internet/Tech'),'BIDU':('China','Internet/Tech'),
    'NTES':('China','Internet/Tech'),'BILI':('China','Internet/Tech'),
    'TME':('China','Internet/Tech'),'ZTO':('China','Internet/Tech'),
    'FUTU':('China','Internet/Tech'),'KWEB':('China','Internet/Tech'),
    'NIO':('China','EV/Auto'),'XPEV':('China','EV/Auto'),
    'LI':('China','EV/Auto'),'TAL':('China','Other'),
    'YUMC':('China','Other'),'MNSO':('China','Other'),
    '0700.HK':('China','HK Blue Chip'),'9988.HK':('China','HK Blue Chip'),
    '3690.HK':('China','HK Blue Chip'),'1810.HK':('China','HK Blue Chip'),
    '2318.HK':('China','HK Blue Chip'),'0941.HK':('China','HK Blue Chip'),
    '1398.HK':('China','HK Blue Chip'),'3988.HK':('China','HK Blue Chip'),
    '0939.HK':('China','HK Blue Chip'),'2628.HK':('China','HK Blue Chip'),
    'ASML':('EU','Tech/Semis'),'SAP':('EU','Tech/Semis'),
    'ERIC':('EU','Tech/Semis'),'NOK':('EU','Tech/Semis'),
    'STM':('EU','Tech/Semis'),'SPOT':('EU','Tech/Semis'),
    'LVMUY':('EU','Luxury'),'HESAY':('EU','Luxury'),
    'PPRUY':('EU','Luxury'),
    'NVO':('EU','Pharma'),'AZN':('EU','Pharma'),
    'GSK':('EU','Pharma'),'SNY':('EU','Pharma'),
    'BAYRY':('EU','Pharma'),
    'HSBC':('EU','Banks'),'UBS':('EU','Banks'),
    'ING':('EU','Banks'),'BBVA':('EU','Banks'),
    'SAN':('EU','Banks'),'DB':('EU','Banks'),
    'TTE':('EU','Energy'),'SHEL':('EU','Energy'),
    'EQNR':('EU','Energy'),'BASFY':('EU','Materials'),
    'SIEGY':('EU','Industrials'),'RELX':('EU','Other'),
    'NSRGY':('EU','Staples'),'UL':('EU','Staples'),
    'DEO':('EU','Staples'),
    'TM':('Japan','Auto'),'HMC':('Japan','Auto'),
    'SONY':('Japan','Tech'),'MUFG':('Japan','Financials'),
    'SMFG':('Japan','Financials'),'NMR':('Japan','Financials'),
    'MFG':('Japan','Financials'),'IX':('Japan','Financials'),
    'DSCSY':('Japan','Pharma'),'NTDOY':('Japan','Gaming'),
    'TSM':('Taiwan','Semis'),'HDB':('India','Banks'),
    'IBN':('India','Banks'),'SHV':('US','Fixed Income'),
}


# =============================================================================
#
#     EVERYTHING BELOW THIS LINE IS THE MODEL ENGINE.
#     IF YOU TOUCH ANYTHING BELOW THIS EVERYTHING WILL EXPLODE
#     SO LIKE, PLEASE DON'T
#
#     The spectral clustering and HRP algorithms are calibrated
#     to work together. Changing one part will break the others.
#
# =============================================================================


def _build_spectral(dist_mat, sigma, knn_k):
    """Gaussian similarity -> kNN graph -> normalized Laplacian -> eigen."""
    n = dist_mat.shape[0]
    W_full = np.exp(-(dist_mat**2) / (2*sigma**2))
    np.fill_diagonal(W_full, 0)
    W = np.zeros_like(W_full)
    for i in range(n):
        d = dist_mat[i].copy(); d[i] = np.inf
        for j in np.argsort(d)[:knn_k]:
            W[i,j] = W_full[i,j]; W[j,i] = W_full[j,i]
    deg = np.where(W.sum(1) > 0, W.sum(1), 1e-10)
    Lrw = np.eye(n) - np.diag(1.0/deg) @ W
    evals, evecs = eigh(Lrw)
    idx = np.argsort(evals)
    return evals[idx], evecs[:,idx], W


def _find_stable_k(dist_mat, knn_k, n_sigma=60):
    """Sweep sigma to find the most stable cluster count."""
    dists = dist_mat[np.triu_indices(dist_mat.shape[0], k=1)]
    sigmas = np.linspace(np.percentile(dists,5), np.percentile(dists,95), n_sigma)
    results = []
    for sig in sigmas:
        ev,_,_ = _build_spectral(dist_mat, sig, knn_k)
        gaps = np.diff(ev[:min(20, len(ev))])
        k_s = np.argmax(gaps[1:])+2 if len(gaps)>1 else 2
        results.append((sig, k_s))
    kc = Counter([r[1] for r in results])
    kb = kc.most_common(1)[0][0]
    ss = [r[0] for r in results if r[1] == kb]
    return kb, np.median(ss), results, kc


def _cap_weights(w, cap):
    """Iteratively cap position weights and redistribute excess."""
    c = w.copy()
    for _ in range(100):
        ex = np.sum(np.maximum(c-cap, 0))
        if ex < 1e-12: break
        m = c < cap; c = np.minimum(c, cap)
        if m.sum() > 0: c[m] += ex*(c[m]/(c[m].sum()+1e-15))
    return c/c.sum()


def _hrp_allocate(cov_mat, ordered_idx):
    """Recursive bisection with inverse-variance weighting."""
    if len(ordered_idx) == 1: return {ordered_idx[0]: 1.0}
    mid = len(ordered_idx)//2
    L, R = ordered_idx[:mid], ordered_idx[mid:]
    lw, rw = _hrp_allocate(cov_mat, L), _hrp_allocate(cov_mat, R)
    def cv(idx, w):
        a = np.array([w[i] for i in idx])
        return a @ cov_mat[np.ix_(idx,idx)] @ a
    lv, rv = cv(L,lw), cv(R,rw)
    a = 1 - lv/(lv+rv+1e-15)
    out = {}
    for i,w in lw.items(): out[i] = a*w
    for i,w in rw.items(): out[i] = (1-a)*w
    return out


def _risk_metrics(w, cov_ann, vols):
    """Portfolio risk analytics."""
    pv = w @ cov_ann @ w; pvol = np.sqrt(max(pv,1e-15))
    hhi = float(np.sum(w**2)); mrc = cov_ann @ w
    return {
        'vol': float(pvol), 'hhi': hhi, 'eff_n': float(1/hhi),
        'div': float((w*vols).sum()/pvol), 'max_w': float(w.max()),
        'rc': (w * mrc / (pv+1e-15)).tolist()
    }


# =============================================================================
#  MAIN PIPELINE
# =============================================================================

def run_optimizer():
    """Run the full pipeline. Returns results dict and prints summary."""

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found in {os.getcwd()}")
        sys.exit(1)

    # --- Load data ---
    prices = pd.read_csv(CSV_PATH, index_col=0, parse_dates=True)
    prices = prices.ffill().dropna()
    prices = prices[prices.index.dayofweek < 5]
    all_tickers = list(prices.columns)

    train_tickers = [t for t in all_tickers if t in TICKER_META]
    n_t = len(train_tickers)
    print(f"  Loaded {n_t} training stocks, {len(prices)} days")

    # --- Returns & market factor removal ---
    train_returns = np.log(prices[train_tickers] / prices[train_tickers].shift(1)).dropna()
    market_ret = train_returns.mean(axis=1)

    residuals = pd.DataFrame(index=train_returns.index, columns=train_tickers, dtype=float)
    betas = {}
    for t in train_tickers:
        y = train_returns[t].values; x = market_ret.values
        b = np.cov(y,x)[0,1] / np.var(x)
        a = np.mean(y) - b*np.mean(x)
        residuals[t] = y - (a + b*x)
        betas[t] = b

    train_corr = residuals.corr().values
    train_corr_raw = train_returns.corr().values

    # --- Spectral clustering ---
    train_dist = np.sqrt(0.5 * (1 - train_corr))
    knn_k = 3

    k_stable, sigma_opt, sigma_results, k_counts = _find_stable_k(train_dist, knn_k)
    eigenvalues, eigenvectors, W_train = _build_spectral(train_dist, sigma_opt, knn_k)

    embed = eigenvectors[:, 1:k_stable+1]
    train_labels = KMeans(n_clusters=k_stable, random_state=42, n_init=30).fit_predict(embed)

    # --- Hierarchical re-clustering of oversized clusters ---
    MAX_FRAC = 0.40
    changed = True
    while changed:
        changed = False
        for c in range(train_labels.max()+1):
            members = np.where(train_labels == c)[0]
            if len(members)/n_t > MAX_FRAC and len(members) > 15:
                sub_dist = train_dist[np.ix_(members, members)]
                sub_knn = max(3, int(np.round(np.log(len(members)))))
                sub_k, sub_sig, _, _ = _find_stable_k(sub_dist, sub_knn)
                if sub_k >= 2:
                    sub_ev, sub_evec, _ = _build_spectral(sub_dist, sub_sig, sub_knn)
                    sub_labels = KMeans(n_clusters=sub_k, random_state=42, n_init=30).fit_predict(sub_evec[:,1:sub_k+1])
                    mx = train_labels.max()
                    for sc in range(sub_k):
                        sub_m = members[sub_labels == sc]
                        train_labels[sub_m] = c if sc == 0 else mx + sc
                    changed = True; break

    # Renumber labels
    unique = np.unique(train_labels)
    lmap = {old:new for new,old in enumerate(unique)}
    train_labels = np.array([lmap[l] for l in train_labels])
    k = len(unique)

    # --- Build cluster info ---
    clusters = []
    for c in range(k):
        midx = [i for i in range(n_t) if train_labels[i] == c]
        secs = Counter([TICKER_META.get(train_tickers[i],('?','?'))[1] for i in midx])
        mkts = Counter([TICKER_META.get(train_tickers[i],('?','?'))[0] for i in midx])
        top_sec = secs.most_common(2)
        label = ' / '.join(s for s,_ in top_sec) if top_sec else '?'
        clusters.append({
            'id': c+1, 'size': len(midx), 'label': label,
            'markets': dict(mkts.most_common()),
            'sectors': dict(secs.most_common()),
            'tickers': [train_tickers[i] for i in midx],
        })

    print(f"  {k} clusters found")

    # --- Map portfolio ---
    port_tickers = [t for t in PORTFOLIO if t in all_tickers]
    port_names = [PORTFOLIO[t][0] for t in port_tickers]
    port_shares = [PORTFOLIO[t][1] for t in port_tickers]
    n_p = len(port_tickers)

    port_ret = np.log(prices[port_tickers] / prices[port_tickers].shift(1)).dropna()
    common = train_returns.index.intersection(port_ret.index)
    train_al = train_returns.loc[common]
    port_al = port_ret.loc[common]
    market_al = market_ret.loc[common]

    # Residuals for portfolio stocks
    port_resid = pd.DataFrame(index=common, columns=port_tickers, dtype=float)
    for t in port_tickers:
        y = port_al[t].values; x = market_al.values
        b = np.cov(y,x)[0,1]/np.var(x)
        a = np.mean(y) - b*np.mean(x)
        port_resid[t] = y - (a+b*x)

    train_resid_al = residuals.loc[common]
    port_labels = np.zeros(n_p, dtype=int)
    port_method = {}
    for i, pt in enumerate(port_tickers):
        if pt in train_tickers:
            port_labels[i] = train_labels[train_tickers.index(pt)]
            port_method[pt] = "direct"
        else:
            corrs = np.array([port_resid[pt].corr(train_resid_al[tt])
                              if tt in train_resid_al.columns else 0.0
                              for tt in train_tickers])
            dists = np.sqrt(0.5*(1-corrs))
            cdists = [np.mean(dists[[j for j in range(n_t) if train_labels[j]==c]])
                      for c in range(k)]
            port_labels[i] = np.argmin(cdists)
            port_method[pt] = "projected"

    # --- HRP allocation ---
    port_corr = port_al[port_tickers].corr().values
    port_cov = port_al[port_tickers].cov().values
    port_vols = port_al[port_tickers].std().values * np.sqrt(252)
    port_cov_ann = port_cov * 252

    port_dist = np.sqrt(0.5*(1-port_corr))
    cond = [port_dist[i,j] for i in range(n_p) for j in range(i+1,n_p)]
    Z = linkage(np.array(cond), method='ward')
    order = leaves_list(Z).tolist()

    hrp_raw = _hrp_allocate(port_cov, order)
    hrp_arr = np.array([hrp_raw[i] for i in range(n_p)])
    hrp_w = _cap_weights(hrp_arr, MAX_WEIGHT)

    last_px = prices[port_tickers].iloc[-1].values
    cur_val = np.array(port_shares) * last_px
    total_val = float(cur_val.sum())
    cur_w = cur_val / total_val
    eq_w = np.ones(n_p) / n_p

    # --- Risk metrics ---
    hrp_m = _risk_metrics(hrp_w, port_cov_ann, port_vols)
    cur_m = _risk_metrics(cur_w, port_cov_ann, port_vols)
    eq_m = _risk_metrics(eq_w, port_cov_ann, port_vols)

    # --- Build output ---
    positions = []
    for i in range(n_p):
        cl = int(port_labels[i])
        positions.append({
            'ticker': port_tickers[i],
            'name': port_names[i],
            'shares': port_shares[i],
            'price': float(last_px[i]),
            'current_value': float(cur_val[i]),
            'current_weight': float(cur_w[i]),
            'hrp_weight': float(hrp_w[i]),
            'target_value': float(hrp_w[i] * total_val),
            'trade': float(hrp_w[i]*total_val - cur_val[i]),
            'cluster': cl + 1,
            'cluster_label': clusters[cl]['label'],
            'vol': float(port_vols[i]),
            'method': port_method[port_tickers[i]],
        })

    # Cluster risk contributions
    cluster_risk = []
    for c in range(k):
        rc_cur = sum(cur_m['rc'][i] for i in range(n_p) if port_labels[i]==c)
        rc_hrp = sum(hrp_m['rc'][i] for i in range(n_p) if port_labels[i]==c)
        rc_eq = sum(eq_m['rc'][i] for i in range(n_p) if port_labels[i]==c)
        n_in = sum(1 for i in range(n_p) if port_labels[i]==c)
        cluster_risk.append({
            'cluster': c+1, 'label': clusters[c]['label'],
            'n_positions': n_in, 'total_stocks': clusters[c]['size'],
            'current_risk': float(rc_cur),
            'hrp_risk': float(rc_hrp),
            'eq_risk': float(rc_eq),
        })

    # Eigenvalue data for technical view
    eigen_data = [{'index': i+1, 'value': float(eigenvalues[i])}
                  for i in range(min(25, n_t))]

    # Correlation matrix for technical view (portfolio only, cluster-sorted)
    psi = sorted(range(n_p), key=lambda i: port_labels[i])
    sorted_corr = port_corr[np.ix_(psi,psi)]

    results = {
        'total_value': total_val,
        'n_training': n_t,
        'n_clusters': k,
        'n_positions': n_p,
        'knn_k': knn_k,
        'sigma': float(sigma_opt),
        'date_range': f"{prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}",
        'max_weight': MAX_WEIGHT,
        'positions': positions,
        'clusters': [c for c in clusters if any(p['cluster']==c['id'] for p in positions)],
        'all_clusters': clusters,
        'cluster_risk': cluster_risk,
        'risk_current': {k:v for k,v in cur_m.items() if k != 'rc'},
        'risk_hrp': {k:v for k,v in hrp_m.items() if k != 'rc'},
        'risk_equal': {k:v for k,v in eq_m.items() if k != 'rc'},
        'eigenvalues': eigen_data,
        'sigma_sweep': [{'sigma': float(r[0]), 'k': int(r[1])} for r in sigma_results],
        'correlation_matrix': {
            'tickers': [port_tickers[i] for i in psi],
            'values': sorted_corr.tolist(),
        },
        'market_factor': {
            'mean_raw_corr': float(train_corr_raw[np.triu_indices(n_t,k=1)].mean()),
            'mean_residual_corr': float(train_corr[np.triu_indices(n_t,k=1)].mean()),
            'variance_explained': float(1 - residuals.var().mean()/train_returns.var().mean()),
        },
    }

    # Pairwise edges for cluster network visualization
    edges = []
    for i in range(n_p):
        for j in range(i+1, n_p):
            edges.append({
                'source': port_tickers[i], 'target': port_tickers[j],
                'corr': float(port_corr[i][j]),
                'same_cluster': int(port_labels[i]) == int(port_labels[j]),
            })
    results['edges'] = edges

    # Save JSON
    with open('portfolio_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    print(f"\n  Portfolio: ${total_val:,.2f} across {n_p} positions")
    print(f"  Clusters: {k} (trained on {n_t} stocks)")
    print(f"  HRP vol: {hrp_m['vol']*100:.2f}% vs current: {cur_m['vol']*100:.2f}%")
    print(f"\n  Saved: portfolio_results.json")
    print(f"  Open portfolio_dashboard.html in your browser")

    return results


if __name__ == '__main__':
    print("=" * 60)
    print("  HRP + Spectral Clustering Portfolio Optimizer")
    print("=" * 60)
    run_optimizer()