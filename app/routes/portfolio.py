from flask import Blueprint, render_template, request, session, jsonify, current_app
from app.utils import plotting_utils
from app.utils.storage_utils import AssetDataManager, PortfolioDataManager
from app.models import PortfolioManager
from app.utils.time_debug import timed
from app.routes.api import _build_assets_payload
import logging
logger = logging.getLogger(__name__)

bp = Blueprint('portfolio', __name__)

@bp.route('/portfolio', methods=['GET', 'POST'])
@bp.route('/', methods=['GET', 'POST'])
@timed
def portfolio_feature():
    app_config = current_app.config['APP_CONFIG']
    data = get_portfolio_data_from_cache()
    
    return render_template(
        'portfolio.html',
        title='Portfolio Dashboard',
        portfolio=data['portfolio'],
        income_plot=data['income_plot'].to_html(full_html=False, include_plotlyjs='cdn')
    )
     
@bp.route('/update_portfolio_data', methods=['POST'])
def update_portfolio_data():
    """Called on UI changes (shares, price, ESG). No market data fetch."""
    data = get_portfolio_data_from_cache()
    portfolio_obj = data['portfolio']

    return jsonify({
        'portfolio': data['portfolio'].to_dict(),
        'income_plot': data['income_plot'].to_json(),
        'assets': _build_assets_payload(portfolio_obj)
    })

@timed
def get_portfolio_data_from_cache():
    """Rebuild portfolio math from cached metrics only."""
    finance_managers = current_app.config['FINANCE_MANAGERS']       
    portfolio_store = current_app.config['PORTFOLIO_STORE'] 
    portfolio = PortfolioManager.from_cache(  
        asset_classes=['stocks','crypto','interest'],
        finance_managers=finance_managers,
        portfolio_store=portfolio_store
    )
    income = portfolio.total_income_data
    # logger.debug(f"[DEBUG] income breakdown: stocks={income.get('stocks')}, "
    #         f"crypto={income.get('crypto')}, interest={income.get('interest')}")
    income_plot = plotting_utils.create_income_plot(portfolio.total_income_data)
    return {'portfolio': portfolio, 'income_plot': income_plot} 

@bp.route('/save_single_value/<asset_type>', methods=['POST'])
def save_single_value(asset_type):
    """
    Handles saving feature update from an AJAX request.
    """
    logger.debug(f"save_single_value called for asset type {asset_type}")
    data = request.get_json()
    ticker = data.get('ticker')
    field = data.get('field')
    value = data.get('value')
    
    if not all([ticker, field]) or value in (None, ''):
        return jsonify({'status': 'error', 'message': 'Invalid data received.'}), 400
        
    # Get the current portfolio state BEFORE the update
    old_data = get_portfolio_data_from_cache()
    old_sub = getattr(old_data['portfolio'], asset_type, None)

    # Snapshot map of ticker_metric -> color string
    old_colors = {}
    if old_sub:
        for asset in old_sub.assets:
            # Using your existing asset.to_dict()['status_colors']
            asset_dict = asset.to_dict() if hasattr(asset, 'to_dict') else {}
            for metric, color in asset_dict.get('status_colors', {}).items():
                old_colors[f"{asset.ticker}_{metric}"] = color

    # Update metric
    manager = AssetDataManager(asset_type)
    clean_value = max(0.0, value) if isinstance(value, (int, float)) else value
    manager.update_ticker_metric(ticker, field, clean_value)
    portfolio_data = get_portfolio_data_from_cache()

    # Get the fresh portfolio state AFTER the update
    new_data = get_portfolio_data_from_cache()
    portfolio_obj = new_data['portfolio']
    new_sub = getattr(portfolio_obj, asset_type, None)

    # Compare status colors to find transitions
    assets_payload = {}
    
    if new_sub:
        for asset in new_sub.assets:
            asset_dict = asset.to_dict() if hasattr(asset, 'to_dict') else {}
            for metric, new_color in asset_dict.get('status_colors', {}).items():
                key = f"{asset.ticker}_{metric}"
                old_color = old_colors.get(key)

                # If the background color actively changed, we have a threshold crossing event!
                if old_color and old_color != new_color:
                    if asset.ticker not in assets_payload:
                        assets_payload[asset.ticker] = {'metrics': {}}
                    
                    # Convert 'bg-green'/'bg-red' to semantic statuses your JS understands
                    def get_status_str(bg_class):
                        if 'green' in bg_class: return 'good'
                        if 'red' in bg_class: return 'bad'
                        return 'caution'

                    # Fetch the actual raw metric value dynamically
                    val = asset.get(metric) if hasattr(asset, 'get') else getattr(asset, metric, 0)

                    assets_payload[asset.ticker]['metrics'][metric] = {
                        'value': val,
                        'old_status': get_status_str(old_color),
                        'new_status': get_status_str(new_color)
                    }

    logger.debug(f"Total threshold updates passing to UI: {assets_payload}")
    logger.debug("save_single_value out")
    return jsonify({
        'status': 'success',
        'portfolio': portfolio_data['portfolio'].to_dict(),
        'income_plot': portfolio_data['income_plot'].to_json(),
        'assets': assets_payload
    })
  
@bp.route('/save_cash', methods=['POST'])
def save_cash():
    data = request.get_json()
    cash_value = data.get('cash', 0)
    
    #storage_utils.save_cash(cash_value)
    #PortfolioDataManager().save_cash(cash_value)
    current_app.config['PORTFOLIO_STORE'].save_cash(cash_value)

    # Store in session so it persists for the user
    session['free_cash'] = float(cash_value)
    
    return jsonify({'status': 'success', 'saved_value': cash_value})
      

@bp.route('/add/<asset_class>', methods=['POST'])
def add_asset(asset_class):
    ticker = request.form.get('ticker', '').upper().strip()
    if not ticker:
        return jsonify({"status": "error", "message": "No ticker provided"}), 400

    if asset_class == 'interest':
        currency = request.form.get('currency', '').upper().strip()
        if not currency:
            return jsonify({"status": "error", "message": "Currency is required"}), 400
        if not ticker:
            ticker = f"{currency}" 
    elif not ticker:
        return jsonify({"status": "error", "message": "No ticker provided"}), 400


    logger.debug(f"[ADD_ASSET] ticker={ticker}, asset_class={asset_class}")
    manager = AssetDataManager(asset_class)
    current_tickers = manager.tickers

    if ticker in current_tickers:
        return jsonify({"status": "exists", "message": f"{ticker} already in portfolio"}), 200

    current_tickers.append(ticker)
    manager.tickers = current_tickers
    logger.info(f"Added {ticker} to {asset_class} portfolio.")

    # Download data for the new ticker before responding
    if asset_class in ('stocks','crypto'):
        finance_manager = current_app.config['FINANCE_MANAGERS'][asset_class]
        live_interval = current_app.config['APP_CONFIG'].get("live_interval")
        research_interval = current_app.config['APP_CONFIG'].get("research_interval")
        
        finance_manager._ensure_prices([ticker], live_interval, force=True)
        if research_interval != live_interval:
            finance_manager._ensure_prices([ticker], research_interval, force=True)

        finance_manager.get_metrics(manager.tickers, interval=live_interval, force=False)
    elif asset_class == 'interest':
        manager.update_ticker_metric(ticker, 'currency', currency)
        manager.update_ticker_metric(ticker, 'compounding', 'annually')  # default, user edits later
    
    return jsonify({"status": "success", "message": f"{ticker} added"}), 200
  
@bp.route('/delete/<asset_class>/<ticker>', methods=['POST'])
def delete_asset(asset_class, ticker):
    manager = AssetDataManager(asset_class)
    manager.delete_ticker_globally(ticker)
    
    # Clean up the external finance manager cache
    fm = current_app.config['FINANCE_MANAGERS'][asset_class]
    live_interval = current_app.config['APP_CONFIG'].get("live_interval")
    research_interval = current_app.config['APP_CONFIG'].get("research_interval")
    fm.remove_ticker(ticker, live_interval)
    fm.remove_ticker(ticker, research_interval)
    
    # Return fresh data
    portfolio_data = get_portfolio_data_from_cache()
    return jsonify({
        'status': 'success',
        'portfolio': portfolio_data['portfolio'].to_dict(),
        'income_plot': portfolio_data['income_plot'].to_json()
    })

# Needed for returning correctly formatted numbers at initialisation
@bp.app_template_filter('format_finance')
def format_finance(val):
    try:
        val = float(val)
        if val == 0: return "0.00"
        if 0 < abs(val) < 0.01: return f"{val:.2e}"
        return f"{val:,.2f}".replace(",", " ")
    except:
        return "N/A"
