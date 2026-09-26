from flask import Blueprint, render_template, request, jsonify, current_app
from app.utils import plotting_utils
from app.models import PortfolioManager
from app.analytics.optimiser import PortfolioOptimiser
from app.analytics.analyser import PortfolioAnalyser
from app.utils.time_debug import timed
import os
import pickle
import pandas as pd
from flask.views import MethodView
from statsmodels.tools import add_constant
from app.analytics.component_registry import ComponentRegistry
from app.analytics.param_sweep import ParamSweepRunner
from app.strategies.cointegration_kalman_class import Config, DataHandler, CointegrationModel, KalmanMLE
import logging
logger = logging.getLogger(__name__)

bp = Blueprint('research', __name__)

# TODO focus on stocks for now, add a general function later
class PortfolioView(MethodView):
    @timed
    def get(self):
        """
        Handle GET requests to /research: render the main portfolio page with initial data.
        """
        dm = current_app.extensions["research_dm"]

        # Load stocks data
        stocks_data = dm.get_data(asset_type='stocks')
        if stocks_data is None or stocks_data.empty:
            return "Error: historical prices could not be loaded.", 404

        # Check for NaNs
        if stocks_data.isnull().values.any():
            current_app.logger.warning("NaN values detected in stocks data")

        # Get sorted tickers
        tickers = sorted(stocks_data.columns.tolist())

        # Determine selected ticker (default first)
        selected_ticker = request.args.get('ticker', tickers[0])

        self._rebuild_analyser(stocks_data)  # side effect isolated here
        analyser = current_app.extensions["portfolio_analyser"]

        return render_template(
            'research.html',
            tickers=analyser.tickers,
            analysis=analyser.to_dict(),
            selected_ticker=selected_ticker,
            title='Research'
        )
    
    @timed
    def _rebuild_analyser(self, stocks_data: pd.DataFrame) -> None:
        finance_managers = current_app.config['FINANCE_MANAGERS']
        portfolio_store = current_app.config['PORTFOLIO_STORE']
        news_manager = current_app.config['NEWS_MANAGER']
        app_config = current_app.config['APP_CONFIG']

        portfolio = PortfolioManager.from_cache(
            asset_classes=['stocks'],
            finance_managers=finance_managers,
            portfolio_store=portfolio_store
        )
        analyser = PortfolioAnalyser(
            portfolio.stocks,
            stocks_data,
            news_manager=news_manager,
            variance_threshold=app_config.get("lsa_variance_threshold"))
        current_app.extensions["portfolio_analyser"] = analyser

class PortfolioDataAPI(MethodView):
    @timed
    def get(self):
        """
        Handle GET requests to /get_data: return JSON plot data for a ticker.
        """
        ticker = request.args.get('ticker')
        ticker2 = request.args.get('ticker2')
        mode = request.args.get('mode', 'price')

        analyser = current_app.extensions.get("portfolio_analyser")
        
        if analyser is None:
            return jsonify({"error": "Portfolio analyser not initialised"}), 500

        asset_analyser = analyser.asset_analysers.get(ticker)

        if asset_analyser is None:
            return jsonify({"error": "Ticker not found"}), 404
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
        if not mode:
            return jsonify({'error': 'No mode provided'}), 400

        ticker_df = asset_analyser.data.to_frame(name=ticker)

        config = {}
        # Prepare plot figure JSON depending on mode
        if mode == 'price':
            config = {'scrollZoom': True}
            fig = plotting_utils.create_price_chart(ticker_df, rolling_windows=[20, 50, 200])
        elif mode == 'returns':
            returns_df = asset_analyser.percent_returns.to_frame(name=ticker)
            fig = plotting_utils.create_returns_distribution_chart(returns_df, asset_analyser.student_t_params)
        elif mode == 'map-2dcorr':
            asset_analyser2 = analyser.asset_analysers.get(ticker2)
            if asset_analyser2 is None:
                return jsonify({"error": "Second ticker not found"}), 404
            
            df1 = asset_analyser.data.to_frame(name=ticker)
            df2 = asset_analyser2.data.to_frame(name=ticker2)

            fig = plotting_utils.create_2d_correlation_map(df1, df2)
        elif mode == 'trend':
            trend_df = asset_analyser.trend.drop(columns=['isPartial'], errors='ignore')
            fig = plotting_utils.create_trends_chart(trend_df, rolling_windows=[7])
        elif mode == 'news':
            try:
                text_analyser = asset_analyser.text_analyser  # store reference to avoid repetition
                if text_analyser is None:
                    return jsonify({
                        "warning": f"Insufficient headlines for {ticker} to run LSA."
                    }), 200
                news_df = text_analyser.lsa()
                logger.info(f"LSA number of components: {len(news_df.columns)}")
                # Intra-cluster similarity — returns a dict of DataFrames
                intra = text_analyser.intra_cluster_similarity()
                for cluster_name, sim_df in intra.items():
                    logger.info(f"\nIntra-cluster similarity for {cluster_name}:\n{sim_df}")

                # Inter-cluster similarity — returns a single DataFrame
                inter = text_analyser.inter_cluster_similarity()
                logger.info(f"\nInter-cluster similarity:\n{inter}")

                # Document similarity — returns a single DataFrame
                doc_sim = text_analyser.document_similarity()
                logger.info(f"\nDocument similarity:\n{doc_sim}")

                jaccard = text_analyser.document_jaccard_similarity()
                logger.info(f"\nJaccard similarity:\n{jaccard}")

                theme = text_analyser.theme_dominance()
                logger.info(f"Themes dominating:\n{theme}")

                lexicon_sentiment = text_analyser.cluster_sentiment()
                finbert_sentiment = text_analyser.cluster_sentiment_finbert()

                if lexicon_sentiment.empty and finbert_sentiment.empty:
                    logger.warning(f"No sentiment signal for {ticker}: insufficient headlines")
                    return jsonify({
                        "warning": f"Insufficient headlines for {ticker} to produce sentiment signal.",
                        "fig_data": None
                    }), 200
                else:
                    logger.info(f"Lexicon sentiment:\n{lexicon_sentiment}")
                    logger.info(f"FinBERT sentiment:\n{finbert_sentiment}")
                fig = plotting_utils.create_lsa_scatter(news_df)
            except Exception as e:
                logger.error(f"[NEWS] LSA failed for {ticker}: {e}")
                return jsonify({"error": f"Could not build news analysis for {ticker}: {str(e)}"}), 500
        else:
            return jsonify({'error': 'Invalid mode'}), 400

        # Convert figure to JSON for frontend rendering
        fig_json = fig.to_json()
        return jsonify({'fig_data': fig_json, 'config': config})

# Register the views with URL rules
bp.add_url_rule('/research', view_func=PortfolioView.as_view('portfolio_research'))
bp.add_url_rule('/get_data', view_func=PortfolioDataAPI.as_view('portfolio_data_api'))


@bp.route('/get_portfolio_data')
@timed
def get_portfolio_data():
    logger.debug("get_portfolio_data called")

    mode = request.args.get('mode', 'returns')
    if not mode:
        return "No mode provided", 400
        
    force_update = request.args.get('force_update') == 'true' # Check for button click

    analyser = current_app.extensions.get("portfolio_analyser")
    if analyser is None:
            return jsonify({"error": "Portfolio analyser not initialised"}), 500
    
    fig = _build_figure(mode, analyser, force_update)

    return jsonify({
            'fig_data': fig.to_json(),
        })

@timed
def _build_figure(mode, analyser, force_update):
    if mode == 'heatmap':
        return plotting_utils.plot_correlation_heatmap(analyser.percent_correlation_matrix)
    elif mode == 'returns':
        return plotting_utils.create_returns_distribution_chart(analyser.weighted_returns)
    elif mode == 'efficient_frontier':
        results = _get_frontier(analyser, force_update)
        return plotting_utils.plot_efficient_frontier_and_portfolios(results, analyser.asset_analysers)
    else:
        raise ValueError(f"Unknown mode: {mode}")

@timed
def _get_frontier(analyser, force_update):
    cache_path = os.path.join(current_app.config['DATA_FOLDER'], "frontier_cache.pkl")
    # Load from cache
    if not force_update and os.path.exists(cache_path):
        logger.info("Loading frontier from cache")
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            current_app.logger.error("Frontier cache load failed, recalculating")
    
    # Perform optimisation
    try:
        opt = PortfolioOptimiser(analyser)
        inputs = analyser.get_optimisation_inputs()
        bounds, constraints = opt.setup_optimisation_constraints(inputs['tickers'], 0.2, True)
        results = opt.perform_full_analysis(bounds, constraints)
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)
        return results
    except Exception as e:
        logger.error(f"Frontier computation failed: {e}")
        raise
            

#@bp.route('/expand_history/<asset_type>', methods=['POST']) #TODO
#def expand_history(asset_type):
@bp.route('/expand_history')
@timed
def expand_history():
    """
        Fetches data on user request.
    """
    # TODO Only for stocks at the moment
    
    # Fetch date inserted by user
    new_start = request.args.get('start')
    if not new_start:
        return jsonify({"message": "Missing start date"}), 400
    logger.debug(f"new_start: {new_start}")

    target_start = pd.to_datetime(new_start)
    finance = current_app.config['FINANCE_MANAGERS']['stocks']
    interval = current_app.config['APP_CONFIG'].get("research_interval")
    tickers = finance._hist_prices.get(interval, pd.DataFrame()).columns.tolist()
    # Trigger the backfill logic
    if not tickers:
        # Fall back to metrics JSON for ticker list
        metrics = finance._load_json(finance._metrics_path, default={})
        tickers = list(metrics.keys())

    try:
        finance._ensure_prices(tickers, interval, force=True, target_start=target_start)
        return jsonify({"message": f"History expanded to {new_start}."})
    except Exception as e:
        current_app.logger.error(f"Expand error: {e}")
        return jsonify({"message": "Failed to expand history."}), 500


@bp.route('/research/components')
def components_schema():
    key = request.args.get('component')
    return jsonify(ComponentRegistry.schema(key) if key else ComponentRegistry.list())

@bp.route('/research/components/sweep', methods=['POST'])
@timed
def run_component_sweep():
    payload = request.get_json()
    dep, indep = payload.get('dep_col'), payload.get('indep_cols', [])
    param_values, mode = payload.get('param_values'), payload.get('mode', 'single')
    entry_z = payload.get('entry_z', 1.0)
    if not (dep and indep and param_values):
        return jsonify({"error": "dep_col, indep_cols, param_values required"}), 400

    assets_data = current_app.extensions["research_dm"].get_data_for_tickers([dep, *indep])
    base_cfg = Config(dep_col=dep, indep_cols=indep)
    data = DataHandler(cfg=base_cfg, df=assets_data[[dep, *indep]], log=base_cfg.log_prices)

    res = CointegrationModel(cfg=base_cfg).fit(data.df.tail(base_cfg.bt_window))
    mle = KalmanMLE(init_beta=res["VECM_beta"][1:], init_alpha=res["VECM_alpha"], cfg=base_cfg)
    w_beta, w_alpha, v = mle.fit(data.df[dep].values,
                                  add_constant(data.df[indep], prepend=False).values)

    runner = ParamSweepRunner(data, base_cfg, res["VECM_beta"][1:], res["VECM_alpha"], w_beta, w_alpha, v)
    df = (runner.sweep_grid(param_values, entry_z) if mode == 'grid'
          else runner.sweep_single(*next(iter(param_values.items())), entry_z=entry_z))
    return jsonify({"results": df.to_dict(orient='records')})