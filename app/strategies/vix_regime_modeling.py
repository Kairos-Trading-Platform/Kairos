import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from numba import njit

# ============================================================
# STEP 1: DATA PREPARATION AND EXPLORATION
# ============================================================

class MarketData:
    """Downloads, cleans and visualizes ETF (TLT/GLD/SPY) and VIX data."""

    def __init__(self, tickers=("TLT", "GLD", "SPY"), vix_ticker="^VIX",
                 start="2000-01-01", end=None):
        self.tickers = list(tickers)
        self.vix_ticker = vix_ticker
        self.start = start
        self.end = end
        self.prices = None     # adjusted close, all assets + VIX
        self.returns = None    # log returns, ETFs only
        self.vix = None        # VIX level
        self.dvix = None       # VIX daily change

    def download(self):
        all_tickers = self.tickers + [self.vix_ticker]
        raw = yf.download(all_tickers, start=self.start, end=self.end,
                           auto_adjust=True, progress=False)["Close"]
        self.prices = raw.dropna(how="any")   # max common sample
        return self

    def compute_returns(self):
        if self.prices is None:
            raise RuntimeError("Call download() first.")
        px = self.prices[self.tickers]
        self.returns = np.log(px / px.shift(1)).dropna()
        self.vix = self.prices[self.vix_ticker]
        self.dvix = self.vix.diff().dropna()

        common_idx = self.returns.index.intersection(self.dvix.index)
        self.returns = self.returns.loc[common_idx]
        self.dvix = self.dvix.loc[common_idx]
        self.vix = self.vix.loc[common_idx]

        self.summary()
        return self

    def summary(self):
        """Displays sample period metrics and observation counts."""
        if self.prices is None or len(self.prices) == 0:
            print("No aligned observations available.")
            return

        start_date = self.prices.index[0].strftime("%Y-%m-%d")
        end_date = self.prices.index[-1].strftime("%Y-%m-%d")
        n_obs = len(self.prices)

        print("=" * 50)
        print("MARKET DATA SUMMARY")
        print("=" * 50)
        print(f"Common sample period : {start_date} to {end_date}")
        print(f"Aligned observations : {n_obs:,}")
        print("=" * 50)

    def run(self):
        return self.download().compute_returns()

    def plot_returns(self):
        self.returns.plot(figsize=(12, 5), title="ETF log returns (TLT / GLD / SPY)")
        plt.ylabel("Log return")
        plt.tight_layout()
        plt.savefig("returns.png")
        plt.close()

    def plot_dvix(self):
        plt.figure(figsize=(12, 5))
        plt.plot(self.dvix.index, self.dvix.values, color="darkorange")
        plt.title(r"$\Delta$VIX")
        plt.ylabel(r"$\Delta$VIX")
        plt.tight_layout()
        plt.savefig("dvix.png")
        plt.close()

# ============================================================
# STEP 2A: DISCRETE MARKOV CHAIN
# ============================================================

@njit(cache=True)
def _count_transitions(states, n_states):
    counts = np.zeros((n_states, n_states))
    for t in range(len(states) - 1):
        counts[states[t], states[t + 1]] += 1.0
    return counts


@njit(cache=True)
def _power_iteration_stationary(P, n_iter=10000, tol=1e-12):
    n = P.shape[0]
    pi = np.full(n, 1.0 / n) # Start with uniform probability distribution
    for _ in range(n_iter):
        pi_new = pi @ P
        diff = 0.0
        for i in range(n):
            diff += abs(pi_new[i] - pi[i]) # Calculate the accumulated difference between all states
        pi = pi_new
        if diff < tol:
            break
    return pi


class MarkovChainVIX:
    """Discrete-state Markov chain on the quantiles of VIX returns."""

    def __init__(self, dvix: pd.Series, n_states: int = 3, labels=None):
        self.dvix = dvix
        self.n_states = n_states
        self.labels = labels or [f"S{i}" for i in range(n_states)]
        self.states_series = None
        self.quantile_edges = None
        self.P = None
        self.stationary = None
        self.log_likelihood_ = None
 
    def fit(self, dvix: pd.Series = None):
        """Fits quantile edges (in-sample) + transition matrix in one call.
        Also serves as the uniform 'fit(y)' entry point expected by
        WalkForwardRegimeModel, mirroring GaussianHMM_VIX.fit()."""
        if dvix is not None:
            self.dvix = dvix
        self.discretise()
        self.fit_transition_matrix()
        return self
 
    def discretise(self, edges=None):
        """If edges is None (default), computes NEW quantile edges from
        self.dvix -- this is the in-sample behaviour used in Steps 1-3.
        If edges IS provided (a frozen array from a training window),
        applies them via pd.cut instead of re-deriving quantiles from
        this data -- this is what makes the model safe to use on
        out-of-sample data in a walk-forward setting, since the bin
        boundaries never see future observations.
        Don't use qcut's labels directly, since _count_transitions
        requires plain integer state codes."""
        if edges is None:
            self.states_series, self.quantile_edges = pd.qcut(
                self.dvix, q=self.n_states, labels=False, retbins=True
            )
        else:
            self.quantile_edges = edges
            self.states_series = pd.cut(
                self.dvix, bins=edges, labels=False, include_lowest=True
            )
        return self
 
    def predict_states(self, dvix_new: pd.Series = None):
        """Assigns states to (optionally new/out-of-sample) DeltaVIX values
        using this model's already-fitted, FROZEN quantile edges -- used
        by WalkForwardRegimeModel so future data never influences the bin
        edges. Values beyond the training range fall in the outermost bin
        (pd.cut already does this via the open-ended outer edges qcut
        produces); any remaining NaN from edge cases is filled from the
        nearest valid neighbour."""
        if self.quantile_edges is None:
            raise RuntimeError("Call fit()/discretise() first.")
        y = self.dvix if dvix_new is None else dvix_new
        states = pd.cut(y, bins=self.quantile_edges, labels=False, include_lowest=True)
        states = states.ffill().bfill().astype(np.int64)
        return states.values

    def predict_states_causal(self, dvix_new: pd.Series = None):
        """Alias of predict_states(): classifying a value against FROZEN,
        fixed bin edges is already pointwise-causal -- each date is
        classified independently of every other date -- so no separate
        causal path is needed here (unlike the HMM, whose Viterbi decode
        is a joint, non-causal optimisation over the whole array)."""
        return self.predict_states(dvix_new)


    def fit_transition_matrix(self):
        if self.states_series is None:
            self.discretise()
        states = self.states_series.values.astype(np.int64)
        counts = _count_transitions(states, self.n_states)
        row_sums = counts.sum(axis=1, keepdims=True)
        if (row_sums == 0).any():  # Check if any row sum is 0
            print("Warning: At least one row_sum is 0. No transitions occurred for some states.")
        row_sums[row_sums == 0] = 1.0  # Fallback to avoid division by zero
        self.P = counts / row_sums # Estimate transition matrix
        return self

    def stationary_distribution(self):
        if self.P is None:
            self.fit_transition_matrix()
        self.stationary = _power_iteration_stationary(self.P)
        return self.stationary

    def plot_regimes(self, vix_level: pd.Series):
        idx = self.states_series.dropna().index
        plt.figure(figsize=(12, 5))
        colors = plt.cm.viridis(np.linspace(0, 1, self.n_states))
        for s in range(self.n_states):
            mask = self.states_series.loc[idx] == s
            plt.scatter(idx[mask], vix_level.loc[idx][mask],
                        color=colors[s], s=8, label=self.labels[s])
        plt.plot(vix_level.loc[idx], color="grey", alpha=0.3, linewidth=0.7)
        plt.title("VIX level coloured by regime (Markov Chain)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("vix_levels_mc.png")
        plt.close()

    def summary(self):
        print("Transition matrix:\n", np.round(self.P, 3))
        print("Stationary distribution:\n", np.round(self.stationary_distribution(), 3))

    def log_likelihood(self):
        """LL of the observed state path under the fitted transition matrix:
        sum over observed transitions i->j of count(i,j) * log P(i,j)."""
        if self.P is None:
            self.fit_transition_matrix()
        states = self.states_series.values.astype(np.int64)
        counts = _count_transitions(states, self.n_states)
        P_safe = np.clip(self.P, 1e-300, 1.0)
        self.log_likelihood_ = float(np.sum(counts * np.log(P_safe)))
        return self.log_likelihood_
 
    def n_parameters(self):
        """Free parameters = transition matrix entries only (n rows, each
        row sums to 1 -> n-1 free entries per row)."""
        return self.n_states * (self.n_states - 1)
 
    def aic(self):
        ll = self.log_likelihood_ if self.log_likelihood_ is not None else self.log_likelihood()
        return 2 * self.n_parameters() - 2 * ll
 
    def bic(self):
        ll = self.log_likelihood_ if self.log_likelihood_ is not None else self.log_likelihood()
        n_transitions = len(self.states_series.dropna()) - 1
        return self.n_parameters() * np.log(n_transitions) - 2 * ll


# ============================================================
# STEP 2B: GAUSSIAN HMM (EM / BAUM-WELCH)
# ============================================================

@njit(cache=True)
def _gaussian_pdf(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2.0 * np.pi))


@njit(cache=True)
def _forward_backward(y, mu, sigma, P, pi):
    # Expectation step of the EM (Baum-Welch) algorithm for a Gaussian HMM.
    #
    # Assumptions:
    #   - Markov property: state_t depends only on state_{t-1}.
    #   - Emissions are Gaussian and depend only on the current state
    #     (conditional independence given the state sequence).
    #   - Model parameters (mu, sigma, P, pi) are fixed/known for this pass
    #     (they get updated afterward in the Maximisation step).
    #
    # Inputs:
    #   y     - (T,) observed sequence (here: daily VIX changes)
    #   mu    - (n,) mean of each state's Gaussian emission
    #   sigma - (n,) std dev of each state's Gaussian emission
    #   P     - (n, n) transition matrix, P[i, j] = P(state_{t+1}=j | state_t=i)
    #   pi    - (n,) initial state distribution, P(state_0 = i)
    #
    # Internal variables:
    #   B[t, i]     - emission likelihood f(y_t | state_t = i), i.e. how
    #                 plausible observation y_t is under state i
    #   alpha[t, i] - forward probability P(state_t = i | y_0...y_t),
    #                 using only past & present observations
    #   c[t]        - normalisation constant at time t (also the scaling
    #                 factor that prevents numerical underflow); its log
    #                 sum gives the total log-likelihood
    #   beta[t, i]  - backward probability, proportional to
    #                 P(y_{t+1}...y_{T-1} | state_t = i), i.e. how well
    #                 state i at time t explains everything that comes after
    #   gamma[t, i] - Smoothed probability P(state_t = i | y_0...y_{T-1}),
    #                 combining alpha and beta -> uses the whole sequence,
    #                 not just the past (this is what makes it "smoothed"
    #                 rather than "filtered")
    #   xi_sum[i,j] - expected number of i -> j transitions over the whole
    #                 sequence, summed over t; used directly in the M-step
    #                 to re-estimate P
    #
    # Output:
    #   gamma, xi_sum, log_likelihood - sufficient statistics for the M-step
    #   plus the log-likelihood of the data under the current parameters
    #   (used to check EM convergence).

    T = len(y)
    n = len(mu)

    # Compute emission probabilities
    B = np.zeros((T, n))
    for t in range(T):
        for i in range(n):
            B[t, i] = max(_gaussian_pdf(y[t], mu[i], sigma[i]), 1e-300)

    # Compute forward probabilities
    alpha = np.zeros((T, n))
    c = np.zeros(T)
    alpha[0] = pi * B[0]
    c[0] = max(alpha[0].sum(), 1e-300)
    alpha[0] /= c[0]
    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ P) * B[t]
        c[t] = max(alpha[t].sum(), 1e-300) # Total probability of observed sequence
        alpha[t] /= c[t]

    # Compute backward probabilities
    beta = np.zeros((T, n))
    beta[T - 1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = (P @ (B[t + 1] * beta[t + 1])) / c[t + 1]

    # Compute smoothed state probabilities (Bayes theorem)
    gamma = alpha * beta
    for t in range(T):
        gamma[t] /= gamma[t].sum()

    # Compute expected transitions
    xi_sum = np.zeros((n, n))
    for t in range(T - 1):
        denom = c[t + 1]
        for i in range(n):
            for j in range(n):
                xi_sum[i, j] += alpha[t, i] * P[i, j] * B[t + 1, j] * beta[t + 1, j] / denom

    # Compute log likelihood (product of c's give the total probability P(Y))
    log_likelihood = np.sum(np.log(c))
    return gamma, xi_sum, log_likelihood

@njit(cache=True)
def _forward_filter(y, mu, sigma, P, pi):
    # Forward-only (filtering) pass: returns alpha[t, i] = P(state_t = i |
    # y_0...y_t), the SAME forward recursion used inside _forward_backward,
    # but computed alone, with no backward pass and no full-sequence
    # Viterbi optimisation. This matters for causality: alpha[t] is
    # mathematically guaranteed to depend only on y[0..t] -- never on any
    # y after t -- unlike:
    #   - gamma (smoothed) in _forward_backward, which also uses beta,
    #     itself computed backward from the END of the array, so gamma[t]
    #     depends on every y after t too;
    #   - _viterbi's decoded path, which finds the single best path over
    #     the WHOLE array at once, so even its state at t is chosen with
    #     knowledge of everything that comes after t in that array.
    # This is the function that must be used to decode states on a block
    # of future/out-of-sample data (see GaussianHMM_VIX.predict_states_causal
    # and WalkForwardRegimeModel) -- passing an array of many future dates
    # to _viterbi or gamma would leak information within that block.
    T = len(y)
    n = len(mu)
    alpha = np.zeros((T, n))
    for t in range(T):
        B_t = np.zeros(n)
        for i in range(n):
            B_t[i] = max(_gaussian_pdf(y[t], mu[i], sigma[i]), 1e-300)
        if t == 0:
            alpha[t] = pi * B_t
        else:
            alpha[t] = (alpha[t - 1] @ P) * B_t
        s = alpha[t].sum()
        if s < 1e-300:
            s = 1e-300
        alpha[t] = alpha[t] / s
    return alpha


@njit(cache=True)
def _viterbi(y, mu, sigma, P, pi):
    # Viterbi decoding: finds the most likely sequence of hidden
    # states given the observations and a fitted model (mu, sigma, P, pi).
    # This is different from gamma in _forward_backward, which gives the
    # marginal probability of each state at each t independently.
    # Viterbi instead finds the single best joint path through all t.
    #
    # Assumptions: same as _forward_backward (first-order Markov states,
    # Gaussian emissions conditional on state). Parameters are already
    # fixed (fitted via EM) when this is called.
    #
    # Works in log-space (log_P, log_pi) to avoid numerical underflow
    # from multiplying many small probabilities together.
    #
    # Internal variables:
    #   delta[t, j] - the highest possible log-probability of any path
    #                 that ends in state j at time t, given y_0...y_t
    #                 (dynamic programming: best score up to t, ending in j)
    #   psi[t, j]   - "backpointer": which state at t-1 achieved that best
    #                 score for delta[t, j] (needed to reconstruct the path)
    #   best_i      - at each t, j: the previous state i that maximizes
    #                 delta[t-1, i] + log_P[i, j], i.e. the most likely
    #                 predecessor of state j
    #   states[t]   - the final decoded path, filled in backward starting
    #                 from the last time step's best state and following
    #                 psi back to t = 0
    #
    # Output:
    #   states - (T,) array of the single most likely hidden state at
    #   each time step (a hard assignment, unlike gamma's soft probabilities)

    T = len(y)
    n = len(mu)
    log_P = np.log(P + 1e-300)
    log_pi = np.log(pi + 1e-300)
    delta = np.zeros((T, n))
    psi = np.zeros((T, n), dtype=np.int64)

    # Initialise first time step. For each state i, compute log-probability of being in it at t=0.
    for i in range(n):
        delta[0, i] = log_pi[i] + np.log(max(_gaussian_pdf(y[0], mu[i], sigma[i]), 1e-300))

    for t in range(1, T): # Time step
        for j in range(n): # State j
            best, best_i = -1e300, 0
            for i in range(n): # Compute log-prob of transitioning from i to j for each possible state i.
                val = delta[t - 1, i] + log_P[i, j]
                if val > best:
                    best, best_i = val, i
            delta[t, j] = best + np.log(max(_gaussian_pdf(y[t], mu[j], sigma[j]), 1e-300)) # Update the prob of observing y[t] given state j
            psi[t, j] = best_i # Record most probable state at time t and state j

    # Reconstruct most likely path
    states = np.zeros(T, dtype=np.int64)
    states[T - 1] = np.argmax(delta[T - 1]) # Select state with highest probability at last time step
    for t in range(T - 2, -1, -1): # Backtrack to find the full path
        states[t] = psi[t + 1, states[t + 1]]
    return states


class GaussianHMM_VIX:
    """Gaussian HMM, fit via EM (Baum-Welch)."""

    def __init__(self, n_states: int = 2, n_iter: int = 200, tol: float = 1e-6,
                 random_state: int = 42, labels: str = None):
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.labels = labels or [f"S{i}" for i in range(n_states)]
        self.rng = np.random.default_rng(random_state)
        self.mu = self.sigma = self.P = self.pi = None
        self.log_likelihood_ = None
        self.gamma_ = None
        self._y = None

    def _init_params(self, y):
        n = self.n_states
        order = np.argsort(y) # Sort the observed data y in ascending order (e.g. y = [3, 1, 4, 2], then order = [1, 3, 0, 2])
        buckets = np.array_split(order, n) # States, e.g. buckets = [array([1, 3]), array([0, 2])]
        mu = np.array([y[b].mean() for b in buckets])
        sigma = np.array([max(y[b].std(), 1e-4) for b in buckets])
        srt = np.argsort(mu)
        mu, sigma = mu[srt], sigma[srt]
        P = np.full((n, n), 1.0 / n)
        np.fill_diagonal(P, 0.9) # Easier to stay in the same state
        P = P / P.sum(axis=1, keepdims=True)
        pi = np.full(n, 1.0 / n)
        return mu, sigma, P, pi

    def fit(self, y: np.ndarray):
        y = np.asarray(y, dtype=np.float64)
        self._y = y
        mu, sigma, P, pi = self._init_params(y)
        ll_prev = -np.inf

        for _ in range(self.n_iter):
            gamma, xi_sum, ll = _forward_backward(y, mu, sigma, P, pi) # E-step

            # Update initial state distribution
            pi = gamma[0].copy()

            # Update transition matrix
            denom = gamma[:-1].sum(axis=0)
            denom[denom == 0] = 1e-300
            P = xi_sum / denom[:, None]
            P = P / P.sum(axis=1, keepdims=True)

            # Update the statistics
            w = gamma.sum(axis=0)
            mu = (gamma * y[:, None]).sum(axis=0) / w # Weighted average
            sigma = np.sqrt((gamma * (y[:, None] - mu) ** 2).sum(axis=0) / w)
            sigma = np.maximum(sigma, 1e-6)

            # Check for convergence
            if abs(ll - ll_prev) < self.tol:
                ll_prev = ll
                break
            ll_prev = ll

        self.mu, self.sigma, self.P, self.pi = mu, sigma, P, pi
        self.log_likelihood_ = ll_prev
        self.gamma_, _, _ = _forward_backward(y, mu, sigma, P, pi)
        return self

    def predict_states(self, y: np.ndarray = None):
        y = self._y if y is None else np.asarray(y, dtype=np.float64)
        return _viterbi(y, self.mu, self.sigma, self.P, self.pi)

    def predict_states_causal(self, y: np.ndarray = None):
        """Filtered (forward-only) state estimate: state_t = argmax_i
        P(state_t = i | y_0...y_t). Unlike predict_states() (Viterbi),
        the estimate at each t never uses y beyond t, so this is safe to
        call on out-of-sample/future data -- this is what
        WalkForwardRegimeModel uses to decode each block."""
        y = self._y if y is None else np.asarray(y, dtype=np.float64)
        alpha = _forward_filter(y, self.mu, self.sigma, self.P, self.pi)
        return np.argmax(alpha, axis=1)


    def sort_states_by_mean(self):
      """Relabel states in ascending order of mu"""
      order = np.argsort(self.mu)
      self.mu = self.mu[order]
      self.sigma = self.sigma[order]
      self.pi = self.pi[order]
      # reorder both rows and columns of P consistently
      self.P = self.P[order][:, order]
      # remap gamma's columns to match the new state order
      self.gamma_ = self.gamma_[:, order]
      return self

    def smoothed_probabilities(self):
        return self.gamma_

    def summary(self):
        print(f"n_states={self.n_states}  log-likelihood={self.log_likelihood_:.2f}")
        print("mu:", np.round(self.mu, 3))
        print("sigma:", np.round(self.sigma, 3))
        print("P:\n", np.round(self.P, 3))

    def plot_regimes(self, index: pd.Index, vix_level: pd.Series):
        states = self.predict_states()
        plt.figure(figsize=(12, 5))
        colors = plt.cm.plasma(np.linspace(0, 1, self.n_states))
        for s in range(self.n_states):
            mask = states == s
            plt.scatter(index[mask], vix_level.loc[index][mask],
                        color=colors[s], s=8, label=self.labels[s])
        plt.plot(vix_level.loc[index], color="grey", alpha=0.3, linewidth=0.7)
        plt.title(f"VIX level coloured by {self.n_states}-state (HMM)")
        plt.legend()
        plt.tight_layout()
        plt.savefig("regimes.png")
        plt.close()
 
    def n_parameters(self):
        """Free parameters: mu (n) + sigma (n) + transition matrix
        (n*(n-1) free entries) + initial distribution (n-1 free entries)."""
        n = self.n_states
        return 2 * n + n * (n - 1) + (n - 1)
 
    def aic(self):
        return 2 * self.n_parameters() - 2 * self.log_likelihood_
 
    def bic(self):
        T = len(self._y)
        return self.n_parameters() * np.log(T) - 2 * self.log_likelihood_

# STEP 3: STATE SELECTION AND INTERPRETATION
 
def compare_models(models: dict) -> pd.DataFrame:
    """Compares fitted MarkovChainVIX / GaussianHMM_VIX models on
    log-likelihood, AIC and BIC. Lower AIC/BIC = better trade-off between
    fit and complexity.
 
    models: dict of {display_name: fitted_model_instance}
    """
    rows = []
    for name, m in models.items():
        ll = getattr(m, "log_likelihood_", None)
        if ll is None:
            ll = m.log_likelihood()
        rows.append({
            "model": name,
            "n_params": m.n_parameters(),
            "log_likelihood": ll,
            "AIC": m.aic(),
            "BIC": m.bic(),
        })
    return pd.DataFrame(rows).set_index("model")
 
 
class RegimeReturnAnalysis:
    """Computes and visualizes ETF return statistics conditional on the
    chosen model's state sequence."""
 
    def __init__(self, returns: pd.DataFrame, states: np.ndarray, index: pd.Index,
                 vix: pd.Series = None, state_labels=None):
        self.returns = returns
        self.states = np.asarray(states)
        self.index = index
        self.vix = vix
        self.n_states = len(np.unique(self.states))
 
        # Sort state IDs by mean VIX level if VIX is provided, so labels
        # are consistent regardless of the model's internal state numbering
        # (e.g. state 0 = Low, state 1 = Medium, state 2 = High)
        if self.vix is not None:
            vix_aligned = self.vix.loc[self.index]
            mean_vix_per_state = [vix_aligned[self.states == s].mean() for s in range(self.n_states)]
            self.state_order = np.argsort(mean_vix_per_state)  # new_rank -> old_state_id
        else:
            self.state_order = np.arange(self.n_states)
 
        self.state_labels = state_labels or [f"State {s}" for s in range(self.n_states)]
        self.stats_ = None
 
    def compute_stats(self):
        """Groups aligned ETF returns by ordered state and computes mean/std."""
        aligned = self.returns.loc[self.index].copy()
        rank_map = {old_s: new_rank for new_rank, old_s in enumerate(self.state_order)}
        aligned["state"] = pd.Series(self.states, index=self.index).map(rank_map)
        grouped = aligned.groupby("state")
        self.stats_ = {"mean": grouped.mean(), "std": grouped.std()}
        return self.stats_
 
    def summary(self):
        if self.stats_ is None:
            self.compute_stats()
        print("--- Ordered State Summary ---")
        for rank, label in enumerate(self.state_labels[:self.n_states]):
            print(f"Rank {rank} ({label}) <- Original Model State ID: {self.state_order[rank]}")
        print("\nMean daily return by state:\n", self.stats_["mean"].round(5))
        print("\nStd. dev. of daily return by state:\n", self.stats_["std"].round(5))
 
    def plot_bar(self):
        """Grouped bar chart: mean and std of ETF returns, one bar group
        per state, side by side for each ETF."""
        if self.stats_ is None:
            self.compute_stats()
 
        mean_df = self.stats_["mean"].T
        std_df = self.stats_["std"].T
        labels_map = {i: self.state_labels[i] for i in range(self.n_states)}
        mean_df = mean_df.rename(columns=labels_map)
        std_df = std_df.rename(columns=labels_map)
 
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        mean_df.plot(kind="bar", ax=axes[0], rot=0, colormap="viridis")
        std_df.plot(kind="bar", ax=axes[1], rot=0, colormap="viridis")
 
        axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
        axes[0].set_title("Mean Daily Return by Regime")
        axes[0].set_ylabel("Mean Log Return")
 
        axes[1].set_title("Return Std. Dev. by Regime")
        axes[1].set_ylabel("Standard Deviation")

        
        plt.tight_layout()
        plt.savefig("etf_returns.png")
        plt.close()
 
# STEP 4: DESIGNING THE ROTATION STRATEGY
 
class RuleBasedRotationStrategy:
    """Rule-based regime rotation: for each state, allocate 100% to the
    ETF with the highest historical mean return in that state.
 
    This class only DEFINES the rule (Step 4). Actual backtesting,
    including the walk-forward and execution-lag bias fixes, is handled
    centrally by BacktestEngine (Step 5) -- see its _walk_forward_backtest()
    method -- so strategies stay easy to add, swap, and compare."""
 
    def __init__(self, returns: pd.DataFrame, states: np.ndarray, index: pd.Index,
                 state_labels=None):
        self.returns = returns
        self.states = np.asarray(states)
        self.index = index
        self.n_states = len(np.unique(self.states))
        self.state_labels = state_labels or [f"State {s}" for s in range(self.n_states)]
        self.allocation_rule_ = None   # dict: state -> chosen ETF (ticker)
 
    def fit_allocation_rule(self):
        """Full-sample rule, for DESIGN/INSPECTION only (Step 4): for each
        state, pick the ETF with the highest historical mean return
        conditional on that state, using the whole sample. This is
        in-sample by construction -- it shows what the rule looks like,
        but BacktestEngine does NOT use it directly to generate returns;
        it re-derives an equivalent rule causally, date by date."""
        aligned = self.returns.loc[self.index].copy()
        aligned["state"] = self.states
        mean_by_state = aligned.groupby("state").mean()
        self.allocation_rule_ = mean_by_state.idxmax(axis=1).to_dict()
        return self.allocation_rule_
 
    def summary(self) -> pd.DataFrame:
        """Returns a small summary table mapping regimes to target ETF allocations."""
        if self.allocation_rule_ is None:
            self.fit_allocation_rule()
 
        table_data = [
            {"Regime": self.state_labels[s] if s < len(self.state_labels) else f"State {s}",
             "Target Allocation": etf}
            for s, etf in self.allocation_rule_.items()
        ]
        return pd.DataFrame(table_data).set_index("Regime")
 
 
# STEP 5: BACKTESTING AND EVALUATION
 
class PerformanceEvaluator:
    """Computes standard performance metrics for a daily return series."""
 
    def __init__(self, returns: pd.Series, periods_per_year: int = 252):
        self.returns = returns.dropna()
        self.ppy = periods_per_year
 
    def cumulative_return(self):
        return (1 + self.returns).prod() - 1
 
    def annualized_return(self):
        n = len(self.returns)
        growth = (1 + self.returns).prod()
        return growth ** (self.ppy / n) - 1 if n > 0 else np.nan
 
    def annualized_volatility(self):
        return self.returns.std() * np.sqrt(self.ppy)
 
    def sharpe_ratio(self, risk_free: float = 0.0):
        excess = self.returns - risk_free / self.ppy
        vol = excess.std()
        return (excess.mean() / vol) * np.sqrt(self.ppy) if vol > 0 else np.nan
 
    def max_drawdown(self):
        cum = (1 + self.returns).cumprod()
        running_max = cum.cummax()
        drawdown = cum / running_max - 1
        return drawdown.min()
 
    def metrics(self) -> dict:
        return {
            "Cumulative Return": self.cumulative_return(),
            "Annualized Return": self.annualized_return(),
            "Annualized Volatility": self.annualized_volatility(),
            "Sharpe Ratio": self.sharpe_ratio(),
            "Max Drawdown": self.max_drawdown(),
        }
 
 
def equal_weight_benchmark(returns: pd.DataFrame) -> pd.Series:
    """1/3-1/3-1/3 portfolio across returns.columns, rebalanced monthly.
    Weights drift with realized returns between rebalance dates and reset
    to equal weight at the start of each new calendar month. Treats daily
    log-returns as approximately additive/multiplicative for portfolio
    weighting purposes (standard small-return simplification)."""
    n_assets = returns.shape[1]
    port_rets = pd.Series(index=returns.index, dtype=float)
    weights = np.full(n_assets, 1.0 / n_assets)
    current_period = None
    for date, row in returns.iterrows():
        period = date.to_period("M")
        if period != current_period:
            weights = np.full(n_assets, 1.0 / n_assets)   # monthly rebalance
            current_period = period
        r = np.asarray(row, dtype=float)
        port_rets.loc[date] = float(np.dot(weights, r))
        weights = weights * (1 + r)
        weights = weights / weights.sum()
    return port_rets
 
 
def performance_table(named_returns: dict, periods_per_year: int = 252) -> pd.DataFrame:
    """named_returns: {display_name: daily return Series}. Returns a
    DataFrame of Cumulative/Annualized Return, Volatility, Sharpe, MDD."""
    rows = {name: PerformanceEvaluator(r, periods_per_year).metrics()
            for name, r in named_returns.items()}
    return pd.DataFrame(rows).T
 
 
def plot_cumulative_comparison(named_returns: dict, title="Cumulative Performance"):
    plt.figure(figsize=(12, 6))
    for name, r in named_returns.items():
        cum = (1 + r.dropna()).cumprod()
        plt.plot(cum.index, cum.values, label=name, linewidth=2)
    
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig("cumulative_performance.png")
    plt.close()
 
class WalkForwardRegimeModel:
    """Wraps a regime model (MarkovChainVIX or GaussianHMM_VIX) and
    re-fits it periodically, using only data available up to each refit
    date. Produces a fully causal state sequence: at any date, the state
    assigned there was decoded using parameters estimated exclusively
    from PRIOR data -- fixing the full-sample in-sample bias that a
    single one-shot fit(dvix) has, one level up from the rotation rule.
 
    model_factory : zero-arg callable returning a fresh, unfitted model,
        e.g. lambda: GaussianHMM_VIX(n_states=3, labels=[...])
          or lambda: MarkovChainVIX(n_states=2, labels=[...])
        Both classes expose a compatible .fit(y) / .predict_states(y)
        interface, so either can be passed here unchanged.
    dvix : full DeltaVIX series (DatetimeIndex) to walk forward over.
    refit_freq : pandas offset alias for how often to re-fit (e.g. "3ME").
    min_train_obs : minimum observations required before the first fit.
    window : "expanding" (all data up to the refit date) or "rolling"
        (only the last window_size observations) -- see the discussion
        on the expanding vs. rolling trade-off.
    window_size : required when window="rolling".
    """
 
    def __init__(self, model_factory, dvix: pd.Series, refit_freq: str = "3ME",
                 min_train_obs: int = 250, window: str = "expanding",
                 window_size: int = None):
        if window not in ("expanding", "rolling"):
            raise ValueError("window must be 'expanding' or 'rolling'")
        if window == "rolling" and window_size is None:
            raise ValueError("window_size is required when window='rolling'")
        self.model_factory = model_factory
        self.dvix = dvix
        self.refit_freq = refit_freq
        self.min_train_obs = min_train_obs
        self.window = window
        self.window_size = window_size
        self.states_ = None    # causal state sequence (pd.Series), set by run()
        self.models_ = []      # [(train_end_date, fitted_model), ...] for inspection
 
    def run(self):
        idx = self.dvix.index
        assert idx.is_monotonic_increasing, "dvix index must be chronologically sorted"

        if len(idx) <= self.min_train_obs:
            raise ValueError("Not enough observations for min_train_obs.")
 
        refit_dates = pd.date_range(idx[self.min_train_obs], idx[-1], freq=self.refit_freq)
        if len(refit_dates) == 0 or refit_dates[0] != idx[self.min_train_obs]:
            refit_dates = pd.DatetimeIndex([idx[self.min_train_obs]]).append(refit_dates)
 
        states = pd.Series(index=idx, dtype=float)
        self.models_ = []
 
        for i, refit_date in enumerate(refit_dates):
            # training window ends at the closest available date <= refit_date
            train_end = idx[idx <= refit_date][-1]
            train = (self.dvix.loc[:train_end] if self.window == "expanding"
                     else self.dvix.loc[:train_end].iloc[-self.window_size:])
            assert train.index.max() <= train_end, \
                "Training window leaked a date past train_end"
 
            model = self.model_factory().fit(train)   # fit on PAST data only
 
            # Relabel states consistently (0 = lowest mean, ... n-1 =
            # highest) across every refit -- EM has no notion of state
            # order, so without this, "state 0" could mean "calm" in one
            # refit and "stressed" in the next, corrupting the combined
            # states_ series used downstream by the rotation strategy.
            if hasattr(model, "sort_states_by_mean"):
                model.sort_states_by_mean()
 
            self.models_.append((train_end, model))

 
            # apply the frozen parameters to the next out-of-sample stretch
            next_date = refit_dates[i + 1] if i + 1 < len(refit_dates) else idx[-1]
            apply_idx = idx[(idx > train_end) & (idx <= next_date)]
            if len(apply_idx) == 0:
                continue

            assert apply_idx.min() > train_end, \
                "Apply window overlaps the training window"

            # predict_states_causal(): forward-filter only (HMM) / frozen
            # bin edges (MC) -- NEVER predict_states()/Viterbi here, since
            # Viterbi optimises jointly over the whole apply_idx block and
            # would let the state at the start of the block see the end
            # of the block (a same-block lookahead bias).

            states.loc[apply_idx] = model.predict_states_causal(self.dvix.loc[apply_idx])
 
        self.states_ = states.dropna().astype(int)
        return self.states_



class BacktestEngine:
    """Manages strategy execution, benchmark generation, metrics calculation,
    and visual comparison across multiple regime models.
 
    Centralizing this here (rather than on RuleBasedRotationStrategy itself)
    is what makes it easy to add and compare several candidate strategies:
    each model is registered once via add_model_strategy(), and the engine
    handles the walk-forward, bias-free backtest identically for all of them.
    """
 
    def __init__(self, returns: pd.DataFrame, dvix_index: pd.Index):
        self.returns = returns
        self.index = dvix_index
        self.strategies = {}           # name -> fitted RuleBasedRotationStrategy
        self.strategies_returns = {}   # name -> backtested daily return Series
        self.benchmarks_returns = {}
 
    def _walk_forward_backtest(self, strategy: "RuleBasedRotationStrategy") -> pd.Series:
        """Walk-forward, bias-free backtest for a fitted RuleBasedRotationStrategy:
 
        1. In-sample bias fix: at each date d, the 'best ETF for this
           state' is computed using an EXPANDING, CAUSAL average -- only
           occurrences of that state on dates strictly before d -- rather
           than strategy.allocation_rule_, which used the whole sample and
           is kept on the strategy only for Step 4 design/inspection.
        2. Execution-lag fix: the resulting choice, which additionally
           requires knowing date d's own state (only known at the close
           of day d), is shifted forward by 1 day before being applied --
           so day d's return is realized using a decision that was fully
           knowable at the close of day d-1.
        """
        aligned = strategy.returns.loc[strategy.index].copy()
        aligned["state"] = strategy.states
        etfs = list(strategy.returns.columns)

        assert strategy.index.is_monotonic_increasing, \
            "Strategy index must be chronologically sorted for causal (expanding) computation"

 
        expanding_mean = pd.DataFrame(index=strategy.index, columns=etfs, dtype=float)
        for s in np.unique(strategy.states):
            mask = aligned["state"].values == s
            sub = aligned.loc[mask, etfs]
            expanding_mean.loc[mask, etfs] = sub.expanding().mean().shift(1).values
 
        all_nan = expanding_mean.isna().all(axis=1)
        causal_choice = expanding_mean.fillna(-np.inf).idxmax(axis=1)
        causal_choice[all_nan] = np.nan
 
        positions = causal_choice.shift(1)   # 1-day execution lag

        # Sanity check that the execution lag was actually applied: if it
        # was (and not accidentally dropped in a future edit), the first
        # date can never have a position (nothing is knowable before the
        # very first observation), and positions must equal causal_choice
        # shifted by exactly one row.
        assert pd.isna(positions.iloc[0]), \
            "Execution lag missing: the first date must have no position"
        assert positions.iloc[1:].reset_index(drop=True).equals(
            causal_choice.iloc[:-1].reset_index(drop=True)
        ), "positions must be causal_choice shifted forward by exactly 1 day"


        aligned_returns = strategy.returns.loc[strategy.index]
        strat_rets = pd.Series(
            [aligned_returns.loc[d, e] if pd.notna(e) else np.nan
             for d, e in positions.items()],
            index=strategy.index
        )
        return strat_rets.dropna()
 
    def add_model_strategy(self, name: str, model, labels=None):
        """Extracts states from a fitted model, fits a rotation strategy,
        and stores its walk-forward backtested daily return series.
 
        `model` can be:
          - a pre-fitted MarkovChainVIX (has .states_series), fit once on
            the full sample -- the regime-model-level in-sample caveat
            still applies to these.
          - a pre-fitted GaussianHMM_VIX (has .predict_states()), same
            caveat.
          - a WalkForwardRegimeModel, which is already fully causal at
            the regime-model level (periodically re-fit on data available
            up to each point) -- this is the bias-free option.
        """
        if isinstance(model, WalkForwardRegimeModel):
            if model.states_ is None:
                model.run()
            states = model.states_.values
            index = model.states_.index          # may start later than self.index
        elif hasattr(model, "states_series") and model.states_series is not None:
            states = model.states_series.values.astype(int)
            index = self.index
        else:
            states = model.predict_states()
            index = self.index
 
        n_states = len(np.unique(states))
        state_labels = labels or (["Low", "High"] if n_states == 2 else ["Low", "Medium", "High"])
 
        strat = RuleBasedRotationStrategy(
            returns=self.returns, states=states,
            index=index, state_labels=state_labels
        )
        strat.fit_allocation_rule()   # Step 4: design/inspection rule only
 
        self.strategies[name] = strat
        self.strategies_returns[name] = self._walk_forward_backtest(strat)
        return self

 
    def add_benchmark(self, name: str, return_series: pd.Series):
        """Adds a reference benchmark return series."""
        self.benchmarks_returns[name] = return_series.loc[self.index].dropna()
        return self
 
    def performance_table(self, periods_per_year: int = 252) -> pd.DataFrame:
        """Returns a DataFrame summarizing metrics across all strategy models and benchmarks."""
        all_series = {**self.strategies_returns, **self.benchmarks_returns}
        return performance_table(all_series, periods_per_year=periods_per_year)
 
    def plot_comparison(self, chosen_model: str = None, title: str = "Cumulative Performance"):
        """Plots cumulative growth for all strategy candidates and benchmarks,
        highlighting the chosen model line."""
        all_series = {**self.strategies_returns, **self.benchmarks_returns}
 
        plt.figure(figsize=(12, 6))
        candidate_colors = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78"]
        cand_idx = 0
 
        for name, r in all_series.items():
            cum = (1 + r.dropna()).cumprod()
 
            if name == chosen_model:
                plt.plot(cum.index, cum.values, label=f"{name} (Chosen)",
                          color="#d62728", linewidth=2.5, zorder=5)
            elif name in self.benchmarks_returns:
                plt.plot(cum.index, cum.values, label=name, color="grey",
                          linestyle="--", alpha=0.7, linewidth=1.2, zorder=2)
            else:
                color = candidate_colors[cand_idx % len(candidate_colors)]
                plt.plot(cum.index, cum.values, label=name, color=color,
                          alpha=0.6, linewidth=1.2, zorder=3)
                cand_idx += 1

        
        plt.title(title)
        plt.ylabel("Cumulative Growth")
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig("plot_comparison.png")
        plt.close()
 
 
# ============================================================
# EXAMPLE USAGE (not executed automatically)
# ============================================================
 
if __name__ == "__main__":
    data = MarketData(start="2000-01-01").run()
    data.plot_returns()
    data.plot_dvix()
 
    # # Markov chain (2 states: low/high)
    # mc2 = MarkovChainVIX(data.dvix, n_states=2, labels=["Low", "High"])
    # mc2.fit_transition_matrix()
    # mc2.summary()
    # mc2.plot_regimes(data.vix)
 
    # # Markov chain (3 states: low/medium/high)
    # mc3 = MarkovChainVIX(data.dvix, n_states=3, labels=["Low", "Medium", "High"])
    # mc3.fit_transition_matrix()
    # mc3.summary()
    # mc3.plot_regimes(data.vix)
 
    # # Gaussian HMM (2 states)
    # hmm2 = GaussianHMM_VIX(n_states=2, labels=["Low", "High"]).fit(data.dvix.values)
    # hmm2.sort_states_by_mean()
    # hmm2.summary()
    # hmm2.plot_regimes(data.dvix.index, data.vix)
 
    # # Gaussian HMM (3 states)
    # hmm3 = GaussianHMM_VIX(n_states=3, labels=["Low", "Medium", "High"]).fit(data.dvix.values)
    # hmm3.sort_states_by_mean()
    # hmm3.summary()
    # hmm3.plot_regimes(data.dvix.index, data.vix)
 
    # # Step 3.1: compare candidate models
    # comparison = compare_models({
    #     "MC (2-state)": mc2,
    #     "MC (3-state)": mc3,
    #     "HMM (2-state)": hmm2,
    #     "HMM (3-state)": hmm3,
    #     })
    # print(comparison)
 
    # # Step 3.2: ETF return stats by regime (state IDs ordered by mean VIX)
    # chosen_states = hmm3.predict_states()
    # regime_analysis = RegimeReturnAnalysis(
    #     returns=data.returns, states=chosen_states, index=data.dvix.index,
    #     vix=data.vix, state_labels=["Low", "Medium", "High"]
    # )
    # regime_analysis.summary()
    # regime_analysis.plot_bar()
 
    # # Step 4: designing the rotation rule (inspection only, per model)
    # hmm3_strategy = RuleBasedRotationStrategy(
    #     returns=data.returns, states=chosen_states,
    #     index=data.dvix.index, state_labels=["Low", "Medium", "High"]
    # )
    # print(hmm3_strategy.summary())
 
    # Step 5: backtest vs. benchmarks
    engine = BacktestEngine(returns=data.returns, dvix_index=data.dvix.index)
 
    # Register all candidate models
    # engine.add_model_strategy("MC (2-state)", mc2)
    # engine.add_model_strategy("MC (3-state)", mc3)
    # engine.add_model_strategy("HMM (2-state)", hmm2)
    # engine.add_model_strategy("HMM (3-state)", hmm3)

    # Walk-forward, fully causal HMM (re-fit every X months, expanding
    # window): fixes the regime-model-level in-sample bias that the
    # static hmm2/hmm3 above still have
    hmm3_wf = WalkForwardRegimeModel(
        model_factory=lambda: GaussianHMM_VIX(n_states=3, labels=["Low", "Medium", "High"]),
        dvix=data.dvix, refit_freq="3ME", min_train_obs=500, window="expanding"
    )
    engine.add_model_strategy("HMM (3-state, walk-forward)", hmm3_wf,
                               labels=["Low", "Medium", "High"])

    hmm2_wf = WalkForwardRegimeModel(
            model_factory=lambda: GaussianHMM_VIX(n_states=2, labels=["Low", "High"]),
            dvix=data.dvix, refit_freq="3ME", min_train_obs=500, window="expanding"
        )
    engine.add_model_strategy("HMM (2-state, walk-forward)", hmm2_wf,
                                labels=["Low", "High"])

 
    # Register benchmarks
    engine.add_benchmark("Equal-Weight (monthly rebal.)", equal_weight_benchmark(data.returns))
    engine.add_benchmark("SPY Buy & Hold", data.returns["SPY"])
 
    # Performance evaluation and plotting
    print("\n--- Performance Metrics ---")
    print(engine.performance_table().round(4))
 
    engine.plot_comparison(
        chosen_model="HMM (3-state, walk-forward)",
        title="Regime rotation strategies vs. benchmarks"
    )