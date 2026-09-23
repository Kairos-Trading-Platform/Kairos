from flask import Flask
from app.utils import database
from app.utils.config import AppConfig
from app.utils.finance_data import FinanceDataManager
from app.services.data_fetching import ResearchDataManager
from app.utils.news_data import NewsDataManager
from app.utils.storage_utils import PortfolioDataManager
from config import Config
import logging

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Set log level based on environment
    log_level = logging.DEBUG if app.config.get('DEBUG') else logging.WARNING

    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )

    ### Silence noisy modules individually
    logging.getLogger('werkzeug').setLevel(logging.WARNING)  # HTTP request logs
    # Routes
    logging.getLogger('app.routes.cashflow').setLevel(logging.WARNING)
    # Utils
    logging.getLogger('app.utils.database').setLevel(logging.WARNING)
    logging.getLogger('app.utils.storage_utils').setLevel(logging.WARNING)
    logging.getLogger('app.utils.time_debug').setLevel(logging.WARNING)
    # yfinance internal loggers
    logging.getLogger('peewee').setLevel(logging.WARNING)
    logging.getLogger('yfinance').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)  # HTTP connection pool logs
    logging.getLogger('asyncio').setLevel(logging.WARNING)  # async internals
    # Numba internal logger
    # yfinance internal loggers
    logging.getLogger('numba').setLevel(logging.WARNING)

    ### Keep these verbose during development
    logging.getLogger('app.utils.finance_data').setLevel(logging.INFO)
    logging.getLogger('app.routes.portfolio').setLevel(logging.DEBUG)
    logging.getLogger('app.models').setLevel(logging.DEBUG)
    

    # User config (intervals, CAGR years, etc.)
    app_config = AppConfig(app.config['USER_CONFIG_PATH'])
    app.config['APP_CONFIG'] = app_config   # attach to app for access in routes

    portfolio_store = PortfolioDataManager(app.config['DATA_FOLDER'])
    app.config['PORTFOLIO_STORE'] = portfolio_store  # so routes can reuse the same instance

    app.config['FINANCE_MANAGERS'] = {
        'stocks': FinanceDataManager(app.config['DATA_FOLDER'], 'stocks', app_config, portfolio_store),
        'crypto': FinanceDataManager(app.config['DATA_FOLDER'], 'crypto', app_config, portfolio_store),
        'interest': FinanceDataManager(app.config['DATA_FOLDER'], 'interest', app_config, portfolio_store),
    }
    app.extensions["research_dm"] = ResearchDataManager(
        app.config['FINANCE_MANAGERS'],
        app_config
    )
    app.config['NEWS_MANAGER'] = NewsDataManager(
        cache_dir=app.config['DATA_FOLDER'],
        config=app_config
    )

    from .routes import cashflow, portfolio, research, strategies, api
    app.register_blueprint(cashflow.bp)
    app.register_blueprint(portfolio.bp)
    app.register_blueprint(research.bp)
    app.register_blueprint(strategies.bp)
    app.register_blueprint(api.bp)
    
    # Initialise a database    
    with app.app_context():
        database.init_db()

    app.config['TEMPLATES_AUTO_RELOAD'] = True  #TODO CHANGE THIS IN PRODUCTION!
    
    # Register filters with Jinja2
    from .routes.cashflow import datetime_format
    app.jinja_env.filters['strftime'] = datetime_format
    app.jinja_env.filters['interval_to_ms'] = interval_to_ms

    return app

def interval_to_ms(interval: str) -> int:
    """Convert interval string to milliseconds for JS polling."""
    units = {"m": 60, "h": 3600, "d": 86400}
    try:
        value = int(interval[:-1])
        unit = interval[-1]
        return value * units[unit] * 1000
    except (ValueError, KeyError):
        return 15 * 60 * 1000  # fallback: 15 minutes
