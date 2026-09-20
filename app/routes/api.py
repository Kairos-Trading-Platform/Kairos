# app/routes/api.py
from flask import Blueprint, jsonify, current_app, request
from app.utils import plotting_utils
from app.utils.storage_utils import AlertNotificationManager
from app.models import PortfolioManager
import logging

bp = Blueprint('api', __name__)
logger = logging.getLogger(__name__)

@bp.route('/api/background_check', methods=['GET'])
def background_check():
    """
    Unified background sync endpoint — callable from any page.
    Returns fresh portfolio data, SMA crossover signals, and current alerts.
    """
    finance_managers = current_app.config['FINANCE_MANAGERS']
    app_config = current_app.config['APP_CONFIG']
    interval = app_config.get("live_interval")

    # Fetch fresh data (respects staleness — won't download if fresh)
    portfolio = PortfolioManager.from_storage(
        asset_classes=['stocks','crypto','interest'],
        finance_managers=finance_managers,
        interval=interval,
        force_update=False
    )

    # SMA crossover signals — computed from cached price data, no network call
    signals = _compute_sma_signals(finance_managers, interval)

    # Load current persisted alerts
    alert_manager = AlertNotificationManager()
    alerts = alert_manager.get_alerts()

    # Add new signals as alerts if not already present
    for signal in signals:
        if not any(a['id'] == signal['id'] for a in alerts):
            alerts.insert(0, signal)
    
    am = alert_manager.save_alerts(alerts)

    income_plot = plotting_utils.create_income_plot(portfolio.total_income_data)

    return jsonify({
        'portfolio': portfolio.to_dict(),
        'income_plot': income_plot.to_json(),
        'assets': _build_assets_payload(portfolio),
        'alerts': alerts
    })


def _compute_sma_signals(finance_managers, interval) -> list[dict]:
    """
    Check price vs SMA crossovers for all tickers.
    Returns list of alert dicts for any crossovers detected.
    """
    from datetime import datetime
    signals = []
    windows = [20, 50, 200]

    for asset_type, manager in finance_managers.items():
        df = manager._hist_prices.get(interval)
        if df is None or df.empty:
            continue
        
        for ticker in df.columns:
            series = df[ticker].dropna()
            if len(series) < max(windows):
                continue
            
            current_price = series.iloc[-1]
            prev_price = series.iloc[-2]

            for window in windows:
                if len(series) < window:
                    continue
                
                sma_current = series.rolling(window).mean().iloc[-1]
                sma_prev = series.rolling(window).mean().iloc[-2]

                # Detect crossover: price crosses SMA
                crossed_above = prev_price <= sma_prev and current_price > sma_current
                crossed_below = prev_price >= sma_prev and current_price < sma_current

                if crossed_above or crossed_below:
                    direction = 'above' if crossed_above else 'below'
                    signal_id = f"{ticker}_sma{window}_{direction}_{series.index[-1].date()}"
                    signals.append({
                        'id':      signal_id,
                        'time':    datetime.now().strftime('%H:%M:%S'),
                        'title':   f'SMA{window} Crossover',
                        'message': f'{ticker} closed {direction} its {window}-period SMA '
                                   f'({current_price:.2f} vs {sma_current:.2f})',
                        'type':    'crossover',
                        'status':  'unread'
                    })
    
    return signals


def _build_assets_payload(portfolio) -> dict:
    """Build the assets dict that NotificationManager.monitorStatusChanges expects."""
    result = {}
    for asset_type in ['stocks','crypto','interest']:
        sub = getattr(portfolio, asset_type, None)
        if not sub:
            continue
        for asset in sub.assets:
            result[asset.ticker] = {
                'metrics': {
                    metric['id']: {
                        'value':       asset.get(metric['id']),
                        'green_limit': metric.get('green_limit'),
                        'red_limit':   metric.get('red_limit')
                    }
                    for metric in asset.get_schema()
                    if metric.get('green_limit') is not None
                }
            }
    return result

@bp.route('/api/alerts', methods=['GET', 'POST'])
def alerts():
    alert_manager = AlertNotificationManager()
    if request.method == 'POST':
        data = request.get_json()
        alert_manager.save_alerts(data.get('alerts', []))
        return jsonify({'status': 'ok'})
    return jsonify({'alerts': alert_manager.get_alerts()})