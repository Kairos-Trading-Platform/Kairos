# app/routes/strategies.py
from flask import Blueprint, jsonify, request, current_app
from app.strategies.base import StrategyRegistry
from app.utils import plotting_utils

bp = Blueprint('strategies', __name__)

@bp.route('/strategies')
def list_strategies():
    return jsonify(StrategyRegistry.list())

@bp.route('/strategies/<key>/schema')
def strategy_schema(key):
    s = StrategyRegistry.get(key)
    return jsonify([vars(p) for p in s.param_schema()])

@bp.route('/strategies/<key>/run', methods=['POST'])
def run_strategy(key):
    s = StrategyRegistry.get(key)
    params = request.get_json()

    # Get tickers from params if specified
    requested = params.get('dep_col') or params.get('tickers')
    if isinstance(requested, list):
        requested.extend(params.get('indep_cols', []))
    
    available = set(price_history.columns)
    missing = set(requested) - available if requested else set()

    dm = current_app.extensions["research_dm"]
    price_history = dm.get_data(asset_type=params.get('asset_type', 'stocks'))   # reuse existing data source
    
    current_app.logger.info(f"price_history shape: {price_history.shape}")
    current_app.logger.info(f"price_history columns: {price_history.columns.tolist() if not price_history.empty else 'EMPTY'}")
    current_app.logger.info(f"price_history index type: {type(price_history.index)}")
    current_app.logger.info(f"available intervals in _hist_prices: {list(dm.finance_managers['stocks']._hist_prices.keys())}")
    
    result = s.run(price_history, params)
    fig = plotting_utils.create_equity_curve_chart(result.equity_curve)  # new, generic helper
    response = {
        'fig_data': fig.to_json(), 
        'metrics': result.metrics,
        'warning': f"Missing {len(missing)} ticker(s): {', '.join(missing)}" if missing else None
    }
    return jsonify(response)

@bp.route('/portfolio/tickers')
def get_portfolio_tickers():
    """Expose live portfolio tickers for strategy auto-fill."""
    dm = current_app.extensions["research_dm"]
    tickers = dm._portfolio_dm.get_live_tickers() if hasattr(dm, '_portfolio_dm') else []
    return jsonify(tickers)