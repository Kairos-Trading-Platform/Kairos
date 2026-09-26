import pandas as pd
from sklearn.model_selection import train_test_split
import logging # TODO logger
import os
import numpy as np
from config import Config
from walk_forward import WalkForwardValidator
from pipeline import compute_and_save_all_pairs
from kalman import KalmanModel
from koopman import KoopmanRegimeFilter
from backtester import Backtester
from performance import Performance
from quality_models import RandomForestModel, XGBoostModel

cfg = Config(
        price_path  = "stocks_price_history_1d_full.csv",
        dep_col      = "ISP.MI",
        indep_cols   = ["BNP.PA"],
        test_size    = 0.3,
        log_prices= True,
        dist_shape   = "gauss",
        kalman_norm_window  = 252,
        bt_window           = 252,
        label_horizon       = 5,
        stop_loss_mult      = 2.0,
        capital             = 100_000,
        cost_bps            = 5.0,
        replay = True,
        coint_enforce_i1=False,
        coint_enforce_johansen=False,
        coint_enforce_po=False
    )
    #download()

# 1. Load data
df_all   = pd.read_csv(cfg.price_path, index_col=cfg.datetime_col,
                        parse_dates=True).dropna()
selected_columns = [cfg.dep_col] + cfg.indep_cols
df_raw =  df_all[selected_columns]
df_train_raw, df_test = train_test_split(df_raw, test_size=cfg.test_size, random_state=42, shuffle=False)

logging.info(f"df_train_raw: {df_train_raw.index[0]} → {df_train_raw.index[-1]}")
logging.info(f"df_test:      {df_test.index[0]} → {df_test.index[-1]}")

# 2. Compute and save results for all pairs (if they don't exist)
compute_and_save_all_pairs(df_all, cfg=cfg)
exit()
# ------------------------------------------------------------------
# RESEARCH MODE — train filter, save state
# ------------------------------------------------------------------

if not os.path.exists(cfg.state_file):
    print("=" * 60)
    print("RESEARCH MODE")
    print("=" * 60)

    # Prepare data
    logging.debug("\nDataHandler for df_train_raw")
    data_raw = DataHandler(cfg=cfg, df=df_train_raw, log=cfg.log_prices)
    data_raw.summary()

    # Automatically scan all possible windows in increments of 20 days
    match = find_first_significant_window(data_raw.df, cfg, step_size=5)
    if match is None:
        logging.error("No cointegration window found.")
        exit()

    ok = automate_strategy_params(cfg, match)
    if not ok:
        logging.error("Window too short for reliable trading.")
        exit()

    cfg.bt_window = match['window_length']
    df_train      = df_train_raw.tail(cfg.bt_window)
    data          = DataHandler(cfg=cfg, df=df_train, log=cfg.log_prices)
    #exit()

    # Find the co-integration model
    model = CointegrationModel(cfg=cfg)
    res = model.fit(data.df)
    confidence_metrics = model.get_model_confidence(res)
    # Merge them for the report
    res.update({f"CONF_{k}": v for k, v in confidence_metrics.items()})
    summary_df = pd.DataFrame.from_dict(res, orient='index', columns=['Value'])
    print(summary_df)

    # Initial state from VECM
    beta_init = res["VECM_beta"][1:]
    alpha_init = res["VECM_alpha"]

    # Build y / X from DataHandler
    y = data.df[cfg.dep_col].values
    X = add_constant(data.df[cfg.indep_cols], prepend=False).values

    # MLE for noise hyperparameters
    mle = KalmanMLE(init_beta=beta_init, init_alpha=alpha_init, cfg=cfg)
    if cfg.dist_shape == 'student':
        opt_W_beta, opt_W_alpha, opt_V, opt_nu = mle.fit(y, X)
    else:
        opt_W_beta, opt_W_alpha, opt_V = mle.fit(y, X)
    
    # Fit filter
    kf = KalmanModel(W_diag=[opt_W_beta]*(cfg.state_dim-1) + [opt_W_alpha],
                    V=opt_V,
                    cfg=cfg)
    kf.initialise(beta_init, alpha_init)

    if cfg.dist_shape == 'student':
        kf.nu = opt_nu
    else:
        kf.nu = None

    df_history = kf.run_and_report(
        y, X, 
        burn_in=cfg.kalman_burn_in, 
        nu=kf.nu, 
        labels=[*cfg.indep_cols, 'Intercept'],
        dates=data.df.index
    )

    e_s = df_history["spread"].values
    Q_s = df_history["variance_Q"].values
    logging.debug(f"Mean Q:        {Q_s.mean():.6f}")
    logging.debug(f"Mean e²:       {(e_s**2).mean():.6f}")
    logging.info(f"Ratio e²/Q:    {(e_s**2).mean() / Q_s.mean():.4f}")

    df_history["future_spread"] = df_history["spread"].shift(-1) - df_history["spread"]
    # correlation
    logging.debug(df_history[["z_score", "future_spread"]].corr())

    logging.info(f"\nTraining history:")
    logging.info(df_history)
    df_history.to_csv(cfg.hist_file)

    kf.last_date = data.df.index[-1]
    kf.save_state(cfg.state_file)
    logging.info(f"Filter state saved to {cfg.state_file}. Last date: {kf.last_date}")


    # ------------------------------------------------------------------
    # WALK-FORWARD VALIDATION
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WALK-FORWARD VALIDATION")
    print("=" * 60)

    opt_nu_val = opt_nu if cfg.dist_shape == 'student' else None
    wfv = WalkForwardValidator(
        data      = data_raw,
        cfg       = cfg,
        beta_init = beta_init,
        alpha_init= alpha_init,
        opt_W_beta = opt_W_beta,
        opt_W_alpha= opt_W_alpha,
        opt_V      = opt_V,
        opt_nu     = opt_nu_val,
    )
    wf_summary = wfv.run()
    wf_summary.to_csv(os.path.join(cfg.output_dir, "walk_forward_summary.csv"), index=False)
    print(f"\nWalk-forward summary saved → {cfg.output_dir}walk_forward_summary.csv")
    print(f"Mean WF Sharpe: {wf_summary['Sharpe'].mean():.4f} "
            f"± {wf_summary['Sharpe'].std():.4f}")

    # ------------------------------------------------------------------
    # BACKTEST + RF + XGB TRAINING
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("BACKTEST + RF + XGB TRAINING")
    print("=" * 60)

    z_score_train = df_history["z_score"].dropna()
    entry_z = np.percentile(np.abs(z_score_train), cfg.entry_z_percentile)
    kf.entry_z = entry_z   # attach to kf before saving
    kf.exit_z  = cfg.exit_z
    kf.save_state(cfg.state_file)

    # Fresh filter instance for backtest (same hyperparams, re-initialised)
    kf_bt = KalmanModel(W_diag=[opt_W_beta]*(cfg.state_dim-1) + [opt_W_alpha],
                        V=opt_V,
                        cfg=cfg)
    kf_bt.nu = opt_nu if cfg.dist_shape == 'student' else None

    # Instantiate Koopman filter — runs upstream during the backtest
    koopman = KoopmanRegimeFilter(cfg=cfg)

    # Use data_raw to backtest on the entire training set
    # No ML gates here: we need clean labels first
    bt = Backtester(data=data_raw, kf=kf_bt, cfg=cfg, entry_z=entry_z)
    bt_results = bt.run(koopman_filter=koopman)
    bt_results.to_csv(cfg.bt_file)

    logging.debug(f"\nBacktest rows: {len(bt_results)}")

    num_signal = (bt_results[cfg.SIGNAL]!=0).sum()
    logging.debug(f"num_signal: {num_signal}")
    logging.info(f"% triggering: {num_signal/len(bt_results)}")

    # Train performance (raw, pre-filter)
    perf = Performance(cfg=cfg)
    perf.evaluate(results=bt_results, label="train_raw")
    perf.plot(results=bt_results, label="train_raw")

    #Build labels and feature matrix
    labels   = bt.build_labels()
    print(f"Label value counts: {labels.value_counts(dropna=False).to_dict()}")
    print(f"Signal value counts: {bt_results[cfg.SIGNAL].value_counts().to_dict()}")    
    features = bt.get_feature_matrix()
    print(f"\nLabelled trades: {labels.notna().sum()}  "
            f"(profitable: {int(labels.sum())}  "
            f"not: {int((labels == 0).sum())})")

    # --- Random Forest ---
    print("\n--- RF CROSS-VALIDATION ---")
    qm = RandomForestModel(n_estimators=cfg.rf_n_estimators, cfg=cfg)
    qm.train(features, labels)
    qm.save(cfg.qm_file)
    print(f"RF model saved → {cfg.qm_file}")

    # --- XGBoost ---
    print("\n--- XGB CROSS-VALIDATION ---")
    xgb_model = XGBoostModel(cfg=cfg)
    xgb_model.train(features, labels)
    xgb_model.save(cfg.xgb_file)
    print(f"XGB model saved → {cfg.xgb_file}")

    # Apply both filters and save
    filtered_train = bt.apply_quality_filter(qm=qm, xgb_model=xgb_model)
    filtered_train.to_csv(cfg.filtered_file)

    raw_sig      = bt_results[cfg.SIGNAL]
    filtered_sig = filtered_train[cfg.SIGNAL]
    logging.info(f"Train: {(raw_sig != 0).sum()} raw signals, {(filtered_sig != 0).sum()} after ML filter")

    # Filtered performance
    print("\n--- PERFORMANCE: RAW vs RF+XGB FILTERED ---")
    perf.compare(bt=bt, qm=qm, xgb_model=xgb_model)

    # RF-only vs XGB-only comparison for reference
    print("\n--- RF-only filter ---")
    filtered_rf_only  = bt.apply_quality_filter(qm=qm)
    perf.evaluate(results=filtered_rf_only,  label="train_rf_only")

    print("\n--- XGB-only filter ---")
    filtered_xgb_only = bt.apply_quality_filter(xgb_model=xgb_model)
    perf.evaluate(results=filtered_xgb_only, label="train_xgb_only")

    # Save the updated config
    cfg.save(cfg.config_file)
    logging.info(f"Saved updated config to {cfg.config_file}")
else:
    # ------------------------------------------------------------------
    # PRODUCTION MODE — update state with new rows
    # ------------------------------------------------------------------

    print("=" * 60)
    print("PRODUCTION MODE")
    print("=" * 60)

    if Path(cfg.config_file).exists():
        cfg = Config.load(cfg.config_file)
        logging.info(f"Loaded config from {cfg.config_file}")
    else:
        logging.error("Couldn't load config file!")
        exit()

    bt_results = pd.read_csv(cfg.bt_file, index_col=cfg.datetime_col, parse_dates=True)
    filtered_train = pd.read_csv(cfg.filtered_file, index_col=cfg.datetime_col, parse_dates=True)
    # Check what happens after a +1 signal specifically
    long_entries  = bt_results[bt_results[cfg.SIGNAL] == 1]
    short_entries = bt_results[bt_results[cfg.SIGNAL] == -1]

    for H in [1, 2, 3, 5]:
        fwd = bt_results[cfg.PNL_RETURN].rolling(H).sum().shift(-H)
        print(f"H={H} | long mean fwd pnl: {fwd[long_entries.index].mean():.6f} "
            f"| short mean fwd pnl: {fwd[short_entries.index].mean():.6f}")

    # Also check raw z vs next return directly, bypassing signal
    logging.debug(f"\nDirect z_score vs fwd pnl:")
    for H in [1, 2, 3, 5]:
        fwd = bt_results[cfg.PNL_RETURN].rolling(H).sum().shift(-H)
        corr = bt_results[cfg.Z_SCORE].corr(fwd)
        logging.debug(f"H={H}: z_score vs fwd_{H}bar_pnl corr = {corr:.6f}")

    df_history = pd.read_csv(cfg.hist_file, index_col=cfg.datetime_col, parse_dates=True)
    e_s = df_history["spread"].values
    Q_s = df_history["variance_Q"].values
    logging.debug(f"Mean Q:        {Q_s.mean():.6f}")
    logging.debug(f"Mean e²:       {(e_s**2).mean():.6f}")
    logging.info(f"Ratio e²/Q:    {(e_s**2).mean() / Q_s.mean():.4f}")

    qm = None
    if os.path.exists(cfg.qm_file):
        qm = SignalQualityModel.load(cfg=cfg)
        print(f"RF model loaded (threshold={qm.threshold:.2f})")
    else:
        print("No RF model found -- running without RF quality filter.")

    xgb_model = None
    if os.path.exists(cfg.xgb_file):
        xgb_model = XGBoostQualityModel.load(cfg=cfg)
        print(f"XGB model loaded (threshold={xgb_model.threshold:.2f})")
    else:
        print("No XGB model found — running without XGB quality filter.")

    # Koopman filter: always instantiate fresh (stateless, runs on live data)
    koopman = KoopmanRegimeFilter(cfg=cfg)
    print(f"Koopman filter instantiated "
            f"(window={cfg.koopman_window}, n_obs={cfg.koopman_n_obs}, "
            f"eig_thresh={cfg.koopman_eig_thresh})")

    test_data = DataHandler(cfg=cfg, df=df_train_raw, df2=df_test, log=cfg.log_prices)
    test_portfolio = Portfolio(cfg=cfg)
    
    # replay=True: re-runs the full test window without advancing checkpoint
    prod_results = run_production(
        data=test_data, cfg=cfg,
        replay=cfg.replay   # don't advance checkpoint during testing
    )

    if cfg.replay:
        bt_results = pd.read_csv(cfg.bt_file, index_col=cfg.datetime_col, parse_dates=True)
        bt_results["year"] = bt_results.index.year
        yearly = bt_results.groupby("year").apply(lambda g: pd.Series({
            "sharpe":    g[cfg.PNL_RETURN].mean() / g[cfg.PNL_RETURN].std() * np.sqrt(252) if g[cfg.PNL_RETURN].std() > 0 else 0,
            "total_ret": g[cfg.PNL_RETURN].sum(),
            "win_rate":  (g[cfg.PNL_RETURN][g[cfg.SIGNAL] != 0] > 0).mean(),
            "n_trades":  (g[cfg.SIGNAL].diff().abs() > 0).sum(),
            "pct_short": (g[cfg.SIGNAL] == -1).mean(),
            "pct_long":  (g[cfg.SIGNAL] == 1).mean(),
        }))
        logging.debug(yearly.round(4))

        if not prod_results.empty:
            print("\nTest period performance")
            perf = Performance(cfg=cfg)
            perf.evaluate(results=prod_results, label="test")
            perf.plot(results=prod_results, label="test")
            perf.compare_df(bt_results, filtered_train, prod_results, 
                                labels=["train_raw", "train_filtered", "test_filtered"])