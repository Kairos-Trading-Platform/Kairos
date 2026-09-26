import logging # TODO logger
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import norm, t as student_t
import joblib
from .config import Config, StepResult


class KalmanModel:
    """
    Linear Gaussian state-space filter tracking time-varying cointegration
    coefficients.
 
    State:    m_t  = [beta1_t, beta2_t, ..., alpha_t]
    Obs eq:   y_t  = X_t . m_t + eps_t,   eps ~ N(0, V)
    State eq: m_t  = m_{t-1} + eta_t,     eta ~ N(0, W)
    """

    def __init__(self, W_diag=None, V=1.0, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to KalmanModel.")
        self.cfg = cfg
        # W represents how much we allow coefficients to drift per step
        if W_diag is None:
            self.W = np.diag([self.cfg.kalman_W_default_beta] * (self.cfg.state_dim - 1) + [self.cfg.kalman_W_default_alpha])
        else:
            if isinstance(W_diag, list):
                W_diag = np.array(W_diag)
            if W_diag.ndim == 1:
                self.W = np.diag(W_diag)
            else:
                self.W = W_diag

        self.V = V
        self.m = None
        self.C = None
        self.nu = 5.0
        self._innov_history  = []   # rolling buffer for past e values
        self._z_score_history = []  # rolling buffer for past z_score values
        self.last_date = None

        # Bind the step function dynamically based on config
        if self.cfg.kalman_debug:
            self.step = self._step_debug
        else:
            self.step = self._step_production

    def initialise(self, beta_indep: np.ndarray, alpha: float):
        """
        beta_indep : coefficients for the independent variables only
                     (i.e. VECM_beta[1:], already excluding the dep-var coeff)
        alpha      : VECM intercept (sign unchanged from CointegrationModel)
        """

        self.m = np.array([*beta_indep, alpha], dtype=float)
        self.C = np.eye(self.cfg.state_dim) * self.cfg.kalman_C_prior

    def step(self, y_t: float, X_t: np.ndarray) -> StepResult:
        """This will be dynamically overwritten in __init__"""
        pass

    def _step_debug(self, y_t: float, X_t: np.ndarray) -> StepResult:
        """
        y_t : scalar observation (log-price of dependent variable)
        X_t : 1-D array [log(x1), log(x2), ..., 1.0]
 
        Returns (innovation e, forecast variance Q, updated state m)
        """
        # Prediction
        # a_t = m_{t-1}, R_t = C_{t-1} + W
        R = self.C + self.W # predicted covariance

        # Forecast
        f = X_t @ self.m # predicted observation
        Q = X_t @ R @ X_t.T + self.V # innovation variance

        # Update
        e = y_t - f # innovation
        K = (R @ X_t.T) / Q # Kalman gain

        self.m = self.m + K * e
        self.C = (np.eye(self.cfg.state_dim) - np.outer(K, X_t)) @ R

        # update buffer
        z_model = e / np.sqrt(Q)
        self._innov_history.append(e)
        self._z_score_history.append(z_model)
        buf_e = self._innov_history[-self.cfg.kalman_norm_window:]
        buf_z = self._z_score_history[-self.cfg.kalman_norm_window:]
        sigma=0
        z_std=0
        if len(buf_e) >= self.cfg.z_score_min_bars:
            sigma   = np.std(buf_e)
            z_std   = np.std(buf_z)
            z_score = (z_model - np.mean(buf_z)) / z_std if z_std > 1e-10 else np.nan
        else:
            z_score = np.nan

        return StepResult(e=e, Q=Q, m=self.m.copy(), C=self.C.copy(),
                          z_score=z_score,
                          innov_std=sigma, z_std=z_std)
    
    def _step_production(self, y_t: float, X_t: np.ndarray) -> StepResult:
        """
        y_t : scalar observation (log-price of dependent variable)
        X_t : 1-D array [log(x1), log(x2), ..., 1.0]
 
        Returns (innovation e, forecast variance Q, updated state m)
        """
        # Prediction
        # a_t = m_{t-1}, R_t = C_{t-1} + W
        R = self.C + self.W # predicted covariance

        # Forecast
        f = X_t @ self.m # predicted observation
        Q = X_t @ R @ X_t.T + self.V # innovation variance

        # Update
        e = y_t - f # innovation
        K = (R @ X_t.T) / Q # Kalman gain

        self.m = self.m + K * e
        self.C = (np.eye(self.cfg.state_dim) - np.outer(K, X_t)) @ R

        # update buffer
        z_model = e / np.sqrt(Q)
        self._innov_history.append(e)
        self._z_score_history.append(z_model)
        buf_e = self._innov_history[-self.cfg.kalman_norm_window:]
        buf_z = self._z_score_history[-self.cfg.kalman_norm_window:]
        sigma=0
        z_std=0
        if len(buf_e) >= self.cfg.z_score_min_bars:
            sigma   = np.std(buf_e)
            z_std   = np.std(buf_z)
            z_score = (z_model - np.mean(buf_z)) / z_std if z_std > 1e-10 else np.nan
        else:
            z_score = np.nan

        return StepResult(e=e, Q=Q, m=self.m.copy(),
                          z_score=z_score,
                          innov_std=sigma, z_std=z_std)
    
    def run_and_report(self, y: np.ndarray, X: np.ndarray,
                       burn_in: int = None, nu: float = None,
                       labels: list = None, dates=None) -> pd.DataFrame:
        """
        Runs the filter over the full training series, prints diagnostics,
        and returns a tidy DataFrame (post burn-in).
        """
        results = [self.step(y[t], X[t]) for t in range(len(y))]
        
        if burn_in is None:
            burn_in = self.kalman_burn_in   # property getter — already enforced
        else:
            # Enforce minimum even when passed explicitly
            burn_in = max(burn_in, self.cfg.z_score_min_bars)
    
        # Unpack post burn-in
        e_s          = np.array([r.e for r in results[burn_in:]])
        Q_s          = np.array([r.Q for r in results[burn_in:]])
        z_score     = np.array([r.z_score for r in results[burn_in:]])
        m_s          = np.array([r.m for r in results[burn_in:]])
        idx          = (dates[burn_in:] if dates is not None
                        else np.arange(len(e_s)))
        
        # Remove any NaN/Inf rows uniformly
        valid = np.isfinite(z_score)
        e_s, Q_s, m_s = e_s[valid], Q_s[valid], m_s[valid]
        z_score_s = z_score[valid]
        if hasattr(idx, '__getitem__'):
            idx = idx[valid]

        # Diagnostics
        self._print_diagnostics(z_score_s, nu)
        self._plot_results(z_score_s, nu, m_s, idx, labels)

        columns = {
            'z_score': z_score_s,
            'spread': e_s,
            'variance_Q': Q_s,
        }

        # Add state coefficients (beta_1, beta_2, ..., alpha)
        for i in range(self.cfg.state_dim):
            if i < self.cfg.state_dim - 1:
                columns[f'beta_{i+1}'] = m_s[:, i]
            else:
                columns['alpha'] = m_s[:, i]

        return pd.DataFrame(columns, index=idx)

    def _print_diagnostics(self, z, nu):
        print(f"\n--- Z-Score ---")
        logging.info(f"Mean: {z.mean():.5f} | Std: {z.std():.5f}")
        logging.info(f"Skew: {stats.skew(z):.5f} | Kurt: {stats.kurtosis(z):.5f}")
        jb_p = stats.jarque_bera(z).pvalue
        sw_p = stats.shapiro(z[:5000]).pvalue
        logging.info(f"JB p: {jb_p:.5f} | SW p: {sw_p:.5f}")
        logging.info(f"Min/Max: {z.min():.4f} / {z.max():.4f}")
        if nu:
            logging.info(f"Student-t 95% threshold (nu={nu:.2f}): "
                f"{student_t.ppf(0.975, df=nu):.4f}")

    def _plot_results(self, z_score, nu, m_s, idx, labels):
        # Distribution plot
        plt.figure(figsize=(10, 6))
        plt.hist(z_score, bins=50, density=True, alpha=0.6, label=f'Z-score')
        x = np.linspace(z_score.min(), z_score.max(), 100)
        plt.plot(x, stats.norm.pdf(x, 0, 1), 'r--', label='Normal Dist')
        if nu:
            plt.plot(x, stats.t.pdf(x, df=nu), 'b-', lw=2, label=f"Student's t (nu={nu:.2f})")
        plt.title("Innovation distribution")
        plt.legend()
        plt.savefig(self.cfg.innov_file)
        plt.close()

        # Coefficient evolution plot
        lbl = labels or [f'State {i}' for i in range(self.cfg.state_dim)]
        fig, axes = plt.subplots(self.cfg.state_dim, 1, figsize=(12, 4 * self.cfg.state_dim), sharex=True)
        for i in range(self.cfg.state_dim):
            axes[i].plot(idx, m_s[:, i], alpha=0.8)
            axes[i].set_title(f'Evolution of {lbl[i]}')
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.cfg.coeff_file)
        plt.close()

    def save_state(self, filepath="kalman_state.pkl"):
        joblib.dump({
            'm':         self.m,
            'C':         self.C,
            'W':         self.W,
            'V':         self.V,
            'nu':        self.nu,
            'last_date': getattr(self, 'last_date', None),
            'innov_history':   self._innov_history,
            'z_score_history': self._z_score_history,
            'entry_z': getattr(self, 'entry_z', 1.0),
            'exit_z':  getattr(self, 'exit_z',  0.0),
        }, filepath)


    @classmethod
    def load_state(cls, cfg: Config) -> 'KalmanModel':
        data     = joblib.load(cfg.state_file)
        instance = cls(W_diag=np.diag(data['W']), V=data['V'], cfg=cfg)
        instance.m         = data['m']
        instance.C         = data['C']
        instance.nu        = data.get('nu', 5.0)
        instance.last_date = data.get('last_date')
        instance._innov_history   = data.get('innov_history',   [])
        instance._z_score_history = data.get('z_score_history', [])
        instance.entry_z = data.get('entry_z', 1.0)
        instance.exit_z  = data.get('exit_z',  0.0)
        return instance

class KalmanMLE:
    """
    Optimises noise hyperparameters (W_beta, W_alpha, V, nu) by maximising
    the log-likelihood of the Kalman innovations.
    """

    def __init__(self, init_beta: np.ndarray,
                 init_alpha: float, cfg: Config = None):
        if cfg is None:
            raise ValueError("A valid Config instance must be provided to KalmanMLE.")
        self.cfg = cfg
        self.init_beta = init_beta
        self.init_alpha = init_alpha
        

    def _neg_ll(self, log_params, y, X):
        params = np.exp(log_params)

        if self.cfg.dist_shape == "student":
            expected_params = 3 + (self.cfg.state_dim - 1)  # W_beta (state_dim-1), W_alpha, V, df
            if len(params) != expected_params:
                raise ValueError(f"Student-t requires {expected_params} parameters: W_beta_1..{self.cfg.state_dim-1}, W_alpha, V, df")
            W_beta = params[:self.cfg.state_dim - 1]  # First state_dim-1 parameters
            W_alpha, Vk, df = params[self.cfg.state_dim - 1], params[self.cfg.state_dim], params[self.cfg.state_dim + 1]

        elif self.cfg.dist_shape == "gauss":
            expected_params = 2 + (self.cfg.state_dim - 1)  # W_beta (state_dim-1), W_alpha, V
            if len(params) != expected_params:
                raise ValueError(f"Gaussian requires {expected_params} parameters: W_beta_1..{self.cfg.state_dim-1}, W_alpha, V")
            W_beta = params[:self.cfg.state_dim - 1]  # First state_dim-1 parameters
            W_alpha, Vk = params[self.cfg.state_dim - 1], params[self.cfg.state_dim]
            df = None
        else:
            raise ValueError("dist must be 'gauss' or 'student'")
    
        W_diag = np.append(W_beta, W_alpha)
        kf = KalmanModel(W_diag=W_diag,V=Vk, cfg=self.cfg)
        kf.initialise(self.init_beta, self.init_alpha)
        ll = 0.0
        for t in range(len(y)):
            sr = kf.step(y[t], X[t])
            scale = np.sqrt(sr.Q)
            if self.cfg.dist_shape == "student":
                ll += student_t.logpdf(sr.e, df=df, loc=0, scale=scale)
            else:  # gaussian
                ll += norm.logpdf(sr.e, loc=0, scale=scale)
        return -ll

    @staticmethod
    def _v_from_ols(y: np.ndarray, X: np.ndarray) -> float:
        """
        Estimate observation noise V from OLS residual variance.

        Why this works: fitting y = X @ beta_ols + residual with a fixed beta
        gives residuals that capture the full spread noise (OU process realisations
        plus any beta-drift contribution).  Their variance is therefore an upper
        bound on V — the true V can be somewhat smaller because some spread
        variance comes from beta drift — but it is always in the right order of
        magnitude and will never collapse to zero.

        We use the full OLS variance (no shrinkage factor) because:
          1. On short windows the OLS beta estimate absorbs some true noise,
             which slightly under-estimates the residual variance.
          2. A conservative (slightly large) V causes the Kalman gain to be
             slightly smaller, which is the safer direction — it makes the
             filter more stable, not less.
        """
        try:
            beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
            resid    = y - X @ beta_ols
            return float(np.var(resid))
        except Exception:
            return None

    def fit(self, y: np.ndarray, X: np.ndarray):
        """
        Maximise the Kalman log-likelihood over W_beta and W_alpha using
        multi-start L-BFGS-B.

        V handling (mle_fix_V_from_ols=True, the default):
            V is fixed at the OLS residual variance rather than being
            optimised.  Profile-likelihood analysis showed that the likelihood
            for V is monotonically increasing as V → 0 on windows of 100–200
            bars: there is no interior optimum and any unconstrained optimiser
            will collapse V to its lower bound regardless of where that bound
            is set.  This is a fundamental identifiability problem — with short
            data, W and V cannot be separately estimated from the innovations
            alone.  Fixing V from OLS breaks the degeneracy: OLS residual
            variance is a principled, data-grounded estimate that reflects the
            true scale of observation noise without relying on the optimiser.

        V handling (mle_fix_V_from_ols=False):
            V is included in the optimisation with multi-start L-BFGS-B.
            Useful for long windows (>500 bars) where V and W become
            separately identifiable, or for diagnostic comparisons.
        """
        fix_V    = self.cfg.mle_fix_V_from_ols
        V_fixed  = None

        if fix_V:
            V_fixed = self._v_from_ols(y, X)
            if V_fixed is None or V_fixed <= 0:
                logging.warning(
                    "OLS V estimation failed — falling back to optimising V.")
                fix_V   = False
                V_fixed = None
            else:
                # Clamp to the configured feasible region so Config bounds
                # are still respected even in fixed-V mode.
                V_fixed = float(np.clip(V_fixed, self.cfg.mle_V_lo,
                                        self.cfg.mle_V_hi))
                logging.info(f"V fixed from OLS residuals: {V_fixed:.4e}")

        # Build bounds over W parameters only (V excluded when fix_V=True)
        bounds = []
        for _ in range(self.cfg.state_dim - 1):
            bounds.append((np.log(self.cfg.mle_W_beta_lo),
                           np.log(self.cfg.mle_W_beta_hi)))
        bounds.append((np.log(self.cfg.mle_W_alpha_lo),
                       np.log(self.cfg.mle_W_alpha_hi)))
        if not fix_V:
            bounds.append((np.log(self.cfg.mle_V_lo),
                           np.log(self.cfg.mle_V_hi)))
        if self.cfg.dist_shape == 'student':
            bounds.append((np.log(self.cfg.mle_nu_lo),
                           np.log(self.cfg.mle_nu_hi)))

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Wrapper: if V is fixed, inject it before evaluating the likelihood
        if fix_V:
            def _objective(log_w_params, y, X):
                # Reconstruct full param vector: [...W_params..., V_fixed, ...]
                n_w = self.cfg.state_dim          # W_beta_1...W_alpha
                if self.cfg.dist_shape == 'student':
                    full = np.concatenate([log_w_params[:n_w],
                                           [np.log(V_fixed)],
                                           log_w_params[n_w:]])
                else:
                    full = np.concatenate([log_w_params, [np.log(V_fixed)]])
                return self._neg_ll(full, y, X)
        else:
            def _objective(log_params, y, X):
                return self._neg_ll(log_params, y, X)

        # Default starting point (physically motivated)
        n_w = self.cfg.state_dim
        w_defaults = [1e-5] * (n_w - 1) + [1e-4]   # W_beta..W_alpha
        if fix_V:
            x0_base = np.log(w_defaults)
            if self.cfg.dist_shape == 'student':
                x0_base = np.append(x0_base, np.log(5.0))
        else:
            x0_base = np.log(w_defaults + [1e-3])   # include V
            if self.cfg.dist_shape == 'student':
                x0_base = np.append(x0_base, np.log(5.0))

        best_ll = np.inf
        best_x  = x0_base.copy()
        n_starts = max(1, self.cfg.mle_n_starts)

        for start_idx in range(n_starts):
            x0 = x0_base if start_idx == 0 else np.random.uniform(lo, hi)
            try:
                res = minimize(_objective, x0=x0, args=(y, X),
                               bounds=bounds, method='L-BFGS-B')
                if res.fun < best_ll:
                    best_ll = res.fun
                    best_x  = res.x
            except Exception as exc:
                logging.warning(f"MLE start {start_idx+1}/{n_starts} failed: {exc}")

        if best_ll == np.inf:
            logging.warning("All MLE starts failed — returning defaults.")
            best_x = x0_base

        # Reconstruct full opt array: [W_beta_1,...,W_alpha, V, (nu)]
        W_opt = np.exp(best_x[:n_w])
        V_out = V_fixed if fix_V else float(np.exp(best_x[n_w]))

        if not fix_V and V_out <= self.cfg.mle_V_lo * 1.01:
            logging.warning(
                f"MLE V = {V_out:.2e} is at its lower bound "
                f"({self.cfg.mle_V_lo:.2e}). The profile likelihood for V "
                "is monotonically increasing toward zero on this window. "
                "Set mle_fix_V_from_ols=True to resolve this.")

        opt = np.concatenate([W_opt, [V_out]])
        if self.cfg.dist_shape == 'student':
            nu_out = float(np.exp(best_x[-1]))
            opt = np.append(opt, nu_out)

        mode = "V fixed from OLS" if fix_V else f"{n_starts} starts"
        msg = (f"MLE ({mode}, ll={-best_ll:.4f})  "
               f"W_beta={' '.join(f'{W_opt[i]:.2e}' for i in range(n_w-1))}  "
               f"W_alpha={W_opt[-1]:.2e}  V={V_out:.2e}")
        if self.cfg.dist_shape == 'student':
            msg += f"  nu={opt[-1]:.2f}"
        logging.info(msg)
        return opt  # [W_beta_1,...,W_alpha, V] or [..., nu] for Student-t
