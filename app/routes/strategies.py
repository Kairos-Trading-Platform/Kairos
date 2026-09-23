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
    dm = current_app.extensions["research_dm"]
    price_history = dm.get_data(asset_type=params.get('asset_type', 'stocks'))   # reuse existing data source
    result = s.run(price_history, params)
    fig = plotting_utils.create_equity_curve_chart(result.equity_curve)  # new, generic helper
    return jsonify({'fig_data': fig.to_json(), 'metrics': result.metrics})