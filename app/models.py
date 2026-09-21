from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import List, Dict, Any
from app.utils.storage_utils import AssetDataManager, PortfolioDataManager
from app.utils.time_debug import timed
from functools import cached_property
import numpy as np
from numba import njit
from datetime import date, timedelta
import logging
logger = logging.getLogger(__name__)

@njit(cache=True)
def _compound_daily_income(principal: float, annual_rate: float,
                            periods_per_year: int, n_days: int) -> np.ndarray:
    """Numba-jitted daily compounding: income earned on each of n_days days."""
    daily_rate = (1.0 + annual_rate / periods_per_year) ** (periods_per_year / 365.0) - 1.0
    income = np.empty(n_days)
    balance = principal
    for i in range(n_days):
        earned = balance * daily_rate
        income[i] = earned
        balance += earned
    return income


def _projection_horizon_end(today: date) -> date:
    """End of the month before the 1-year anniversary of `today`."""
    try:
        anniversary = today.replace(year=today.year + 1)
    except ValueError:  # Feb 29
        anniversary = today.replace(year=today.year + 1, day=28)
    return anniversary.replace(day=1) - timedelta(days=1)

# class Metric:
#     def __init__(self, id, label, type="finance",
#                  suffix="", prefix="",
#                  good=None, caution=None):

#         self.id = id
#         self.label = label
#         self.type = type
#         self.suffix = suffix
#         self.prefix = prefix

#         self.good = good
#         self.caution = caution

#     def get_status_class(self, value):
#         if self.good and self.good(value):
#             return "bg-green"

#         if self.caution and self.caution(value):
#             return "bg-orange"

#         return "bg-red"

@dataclass
class AssetData:
    ticker: str
    metrics: Dict[str, Any]
    asset_type: str = 'stocks'
    shares: float = 0.0
    price: float = 0.0
    env: int = 0
    soc: int = 0
    gov: int = 0
    cont: int = 0
    weight: float = 0.0
    syield: float = 0.0
    iyield: float = 0.0
    currency_choice: str = ""     # user-picked currency, interest only
    compounding: str = "daily" # daily/monthly/quarterly/annually

    fx_rates: Dict[str, float] = field(default_factory=dict)
    sector: str = field(init=False, default="Null")
    months_paid: List[int] = field(init=False, default_factory=lambda: [0]*12)
    quote: float = field(init=False, default=0.0)
    quote_eur: float = field(init=False, default=0.0)
    latest_div: float = field(init=False, default=0.0)
    div_yield: float = field(init=False, default=0.0)
    div_growth: float = field(init=False, default=0.0)
    pe_ratio: float = field(init=False, default=0.0)
    fwd_pe: float = field(init=False, default=0.0)
    peg: float = field(init=False, default=0.0)
    pb_ratio: float = field(init=False, default=0.0)
    bench_pb: float = field(init=False, default=0.0)
    earnings_gr: float = field(init=False, default=0.0)
    payout_ratio: float = field(init=False, default=0.0)

# Define a single asset (one per ticker)
@dataclass
class Asset(AssetData):
    def __post_init__(self):
        metrics = self.metrics
        self.sector = metrics.get("Sector", "Null")
        self.quote = self._safe_float(metrics.get("Quote"))
        self.quote_eur = self._safe_float(metrics.get("Quote_EUR"))
        self.latest_div = self._safe_float(metrics.get("Latest_Div_EUR"))
        self.months_paid = metrics.get("Months_Paid", [0]*12)
        self.div_yield = self._safe_float(metrics.get("Div_Yield",0)*100)
        self.div_growth = self._safe_float(metrics.get("Div_CAGR",0)*100)
        self.pe_ratio = self._safe_float(metrics.get("P/E", 0))
        self.fwd_pe = self._safe_float(metrics.get("Fwd_P/E", 0))
        self.peg = self._safe_float(metrics.get("PEG", 0))
        self.pb_ratio = self._safe_float(metrics.get('P/B',0))
        self.bench_pb = self._safe_float(metrics.get('Sector_PB_Benchmark',0))
        self.earnings_gr = self._safe_float(metrics.get('Earnings_Growth',0)*100)
        self.payout_ratio = self._safe_float(metrics.get('PayoutRatio',0)*100)
        self.currency = metrics.get("Currency", "USD").lower()

        if self.asset_type == 'interest':
            self.currency = (self.currency_choice or "EUR").lower()
            self.quote = 1.0
            self.quote_eur = self.fx_rates.get(f"{self.currency}_eur_rate", 1.0)

    # Common columns for schemas
    COLUMNS = {
        'ticker':       {'id': 'ticker',       'label': 'Ticker',           'type': 'ticker'},
        'shares':       {'id': 'shares',       'label': 'Shares',           'type': 'input'},
        'price':        {'id': 'price',        'label': 'Avg Price (base)', 'type': 'input'},
        'price_eur':    {'id': 'price_eur',    'label': 'Avg Price (€)',    'type': 'finance', 'suffix': ' €'},
        'market_value': {'id': 'market_value', 'label': 'Value (€)',        'type': 'finance', 'suffix': ' €'},
        'market_value_eur': {'id': 'market_value_eur', 'label': 'Value (€)', 'type': 'finance', 'suffix': ' €'},
        'quote':        {'id': 'quote',        'label': 'Quote',            'type': 'finance'},
        'quote_eur':    {'id': 'quote_eur',    'label': 'Quote (€)',        'type': 'finance', 'suffix': ' €'},
        'weight':       {'id': 'weight',       'label': 'Weight',           'type': 'monitor', 'suffix': '%'},
    }
        
    METRIC_COLOUR_RULES = {
        'stocks':{
            'pe_ratio': {'low': 10,  'high': 20, 'dir': 'low_is_better'},
            'fwd_pe': {'low': 10,  'high': 20, 'dir': 'low_is_better'},
            'earnings_gr': {'low': 0,  'high': 0, 'dir': 'high_is_better'},
            'peg': {'low': 0.9,  'high': 1, 'dir': 'low_is_better'},
            'div_yield': {'low': 2, 'high': 4, 'dir': 'high_is_better'},
            'div_growth': {'low': 4, 'high': 8, 'dir': 'high_is_better'},
            'payout_ratio': {'low': 90, 'high': 100, 'dir': 'low_is_better'},
            'env': {'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'soc': {'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'gov': {'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'cont': {'low': 4,   'high': 6,  'dir': 'high_is_better'},
            'pb_ratio': {'low': 'bench_pb', 'high': 'bench_pb', 'dir': 'low_is_better'},
            'weight': {'low': 4, 'high': 5, 'dir': 'low_is_better'}
        },
        'crypto':{
            'weight': {'low': 40, 'high': 50, 'dir': 'low_is_better'},
            'syield': {'low': 2, 'high': 4, 'dir': 'high_is_better'}
        },
        'interest':{
            'iyield': {'low': 2, 'high': 4, 'dir': 'high_is_better'}
        }
    }

    ANNUAL_COMPOUNDING_PERIODS = {'daily': 365, 'monthly': 12, 'quarterly': 4, 'annually': 1}

    # Ensure floats are retrieved from data
    def _safe_float(self, val):
        try:
            return float(val) if val is not None else 0.0
        except (ValueError, TypeError):
            return 0.0

    @property
    def annual_yield(self) -> float:
        if self.asset_type == 'crypto':
            return self.syield / 100
        if self.asset_type == 'interest':
            return self.iyield / 100
        return 0.0

    @property
    def compounding_periods_per_year(self) -> int:
        if self.asset_type == 'crypto':
            return 365  # staking rewards compound daily
        return self.ANNUAL_COMPOUNDING_PERIODS.get(self.compounding, 365)

    def get_monthly_income(self):
        if self.asset_type in ('crypto', 'interest'):
            return self._projected_monthly_income()
        if not hasattr(self, 'months_paid'):
            return [0.0] * 12
        return [self.latest_div * self.shares if m == 1 else 0.0 for m in self.months_paid]

    def _projected_monthly_income(self) -> list:
        """Compounded daily rewards, bucketed by calendar month, for the next 12 months
        (today -> end of month before the 1-year anniversary)."""
        monthly = [0.0] * 12
        principal = self.market_value_eur
        # logger.debug(f"[DEBUG] {self.ticker} ({self.asset_type}): "
        #         f"shares={self.shares}, quote_eur={self.quote_eur}, "
        #         f"principal={principal}, annual_yield={self.annual_yield}, "
        #         f"syield={self.syield}, iyield={self.iyield}")
        if principal <= 0 or self.annual_yield <= 0:
            return monthly

        today = date.today()
        n_days = (_projection_horizon_end(today) - today).days + 1
        daily_income = _compound_daily_income(
            principal, self.annual_yield, self.compounding_periods_per_year, n_days
        )
        for offset, amount in enumerate(daily_income):
            m = (today + timedelta(days=offset)).month - 1
            monthly[m] += amount
        return monthly  

    @cached_property
    def market_value(self) -> float:
        """
        Calculates nominal market value in base asset currency.
        If price is present, market_value = shares * price.
        If interest/cash (no price), market_value = shares.
        """
        if not self.price:
            return float(self.shares)
        
        return float(self.shares) * float(self.quote)

    @cached_property
    def market_value_eur(self) -> float:
        return float(self.shares) * float(self.quote_eur)
        
    @cached_property
    def cost_basis(self):
        #logger.debug("cost_basis called")
        #logger.debug(f"Asset: {self.ticker}, Price: {self.price_eur}, Shares: {self.shares}")
        return self.shares * self.price_eur
    
    @cached_property
    def price_eur(self):
        """
        Dynamically calculates price in EUR based on the asset's currency.
        """
        # Matches 'usd' with 'usd_eur_rate'
        #logger.debug("price_eur called")
        rate_key = f"{self.currency}_eur_rate"
        rate = self.fx_rates.get(rate_key, 1.0) # Default to 1.0 if already EUR or not found
        #logger.debug(f"Asset: {self.ticker}, Price: {self.price}, Rate: {rate}")
        return self.price * rate
        
    @cached_property
    def asset_income(self):
        return self.market_value * self.div_yield
        
    @cached_property
    def annual_dividend(self):
        payment_frequency = sum(self.months_paid)
        total = self.latest_div * payment_frequency * self.shares
        return total
    
    # Define schema for the HTML renderer
    def get_schema(self):
        # Schema for metricless assets
        if self.asset_type == 'interest':
            return [
                {**self.COLUMNS['ticker'], 'label': 'Name'},
                {'id': 'currency', 'label': 'Currency', 'type': 'text'},
                {**self.COLUMNS['shares'], 'label': 'Amount'},
                {**self.COLUMNS['market_value_eur'], 'label': 'Amount (€)'},
                {**self.COLUMNS['weight']},
                {'id': 'iyield', 'label': 'Interest rate', 'type': 'monitor_input', 'suffix': '%'},
                {'id': 'compounding', 'label': 'Compounding rate', 'type': 'select_input',
                'options': ['daily', 'monthly', 'quarterly', 'annually']},
            ]

        schema = [self.COLUMNS[c] for c in
                  ('ticker', 'shares', 'price', 'price_eur', 'market_value_eur', 'quote', 'quote_eur', 'weight')]

        # Category-specific columns
        if self.asset_type == 'stocks':
            schema += [
                {'id': 'pe_ratio',    'label': 'P/E',         'type': 'monitor'},
                {'id': 'fwd_pe',      'label': 'Fwd P/E',     'type': 'monitor'},
                {'id': 'pb_ratio',    'label': 'P/B',         'type': 'monitor'},
                {'id': 'peg',         'label': 'PEG',         'type': 'monitor'},
                {'id': 'earnings_gr', 'label': 'Earnings gr', 'type': 'monitor', 'suffix': '%'},
                {'id': 'div_yield',   'label': 'Div. yield',  'type': 'monitor', 'suffix': '%'},
                {'id': 'div_growth',  'label': 'Div. CAGR',   'type': 'monitor', 'suffix': '%'},
                {'id': 'payout_ratio','label': 'Payout ratio','type': 'monitor', 'suffix': '%'},
                {'id': 'env',         'label': 'E',           'type': 'monitor_input'},
                {'id': 'soc',         'label': 'S',           'type': 'monitor_input'},
                {'id': 'gov',         'label': 'G',           'type': 'monitor_input'},
                {'id': 'cont',        'label': 'Cont',        'type': 'monitor_input'},
                {'id': 'latest_div',  'label': 'Latest div.', 'type': 'finance', 'suffix': ' €'},
                {'id': 'months_paid', 'label': 'Months paid', 'type': 'visualizer'},
                {'id': 'annual_dividend','label': 'Annual div','type': 'finance', 'suffix': ' €'},
                {'id': 'currency',    'label': 'Curr',        'type': 'text'},
                {'id': 'sector',      'label': 'Sector',      'type': 'text'},
            ]
        elif self.asset_type == 'crypto':
            schema += [
                {'id': 'syield', 'label': 'Staking yield', 'type': 'monitor_input', 'suffix': '%'}
            ]
        else:
            schema += [
                {'id': 'iyield', 'label': 'Interest yield', 'type': 'monitor_input', 'suffix': '%'}
            ]
        return schema
    
    def get_metric_config(self, metric_name):
        """Wrapper to call get_metric_config."""
        return get_metric_config(self, metric_name, self.METRIC_COLOUR_RULES, self.asset_type)
    
    # Convert to a dictionary for the frontend
    def to_dict(self):
        # Basic attributes
        data = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        # Properties
        data.update({
            'market_value': self.market_value,
            'market_value_eur': self.market_value_eur,
            'price_eur': self.price_eur,
            'cost_basis': self.cost_basis,
            'annual_dividend': self.annual_dividend,
            'asset_income': self.asset_income,
            'status_colors': {
                m: self.get_metric_config(m)['class']
                for m in self.METRIC_COLOUR_RULES.get(self.asset_type, {}).keys()
            },
            'schema': self.get_schema()
        })
        return data
        
    def __repr__(self):
        return f"Asset({self.ticker}, MarketValue={self.market_value:.2f})"
        
# Define a portfolio of a given asset type (stocks, crypto, etc...)        
class Portfolio:
    def __init__(self, assets: List[Asset]):
        self.assets = assets
        self.update_weights()

    @property
    def total_market_value(self):
        return sum(asset.market_value_eur for asset in self.assets)
        
    @property
    def total_cost_basis(self):
        return sum(asset.cost_basis for asset in self.assets)
        
    @property
    def monthly_income_data(self):
        payouts = [0.0]*12
        counts = [0] * 12
        details = [[] for _ in range(12)] # List of lists to store stock details per month
        
        for asset in self.assets:
            asset_payouts = asset.get_monthly_income()
            for i in range(12):
                if asset_payouts[i] > 0:
                    payouts[i] += asset_payouts[i]
                    counts[i] += 1
                    details[i].append(f"{asset.ticker}: €{asset_payouts[i]:.2f}") # Details for hovering in plot

        # Process details to include the total income in plot
        final_details = []
        for i in range(12):
            if payouts[i] > 0:
                # Combine the ticker list and add the total header
                ticker_list = "<br>".join(details[i])
                final_details.append(f"<b>Total: €{payouts[i]:.2f}</b><br><br>{ticker_list}")
            else:
                final_details.append("No Income")
                
        return SimpleNamespace(
            payouts = payouts,
            counts = counts,
            details = final_details
        )

    @property
    def annual_dividends(self):
        return sum(self.monthly_income_data.payouts)
    
    @property
    def portfolio_yield_data(self):
        total_mv = self.total_market_value
        total_income = sum(asset.market_value_eur * asset.div_yield for asset in self.assets)
        total_income_growth = sum(asset.market_value_eur * asset.div_yield * asset.div_growth for asset in self.assets)
        
        if total_mv < 0 or total_income <= 0:
            return SimpleNamespace(
            div_yield = 0,
            div_growth= 0
        )

        return SimpleNamespace(
            div_yield = float(total_income / total_mv),
            div_growth = float(total_income_growth / total_income)
        )
    
    @property
    def portfolio_syield(self):
        total_mv = self.total_market_value
        total_income = sum(asset.market_value_eur * asset.syield for asset in self.assets)
        if total_mv > 0:
            return float(total_income / total_mv)

    @property
    def portfolio_iyield(self):
        total_mv = self.total_market_value
        total_income = sum(asset.market_value_eur * asset.iyield for asset in self.assets)
        if total_mv > 0:
            return float(total_income / total_mv)
 
    @property
    def sectors(self):
        sectors = {}
        for asset in self.assets:
            sector = asset.sector
            sectors[sector] = sectors.get(sector,0) + asset.market_value
            
        return SimpleNamespace(
            labels = list(sectors.keys()),
            values = list(sectors.values())
        )

    def update_weights(self):
        total_mv = self.total_market_value
        if self.total_market_value > 0:
            for asset in self.assets:
                asset.weight = (asset.market_value_eur / total_mv) * 100
        else:
            for asset in self.assets:
                asset.weight = 0

    METRIC_COLOUR_RULES = {
        'stocks':{
            'div_yield': {'low': 2, 'high': 4, 'dir': 'high_is_better'},
            'div_growth': {'low': 4, 'high': 8, 'dir': 'high_is_better'},
            },
        'crypto':{
            'syield': {'low': 2, 'high': 4, 'dir': 'high_is_better'}
        },
        'interest':{
            'iyield': {'low': 2, 'high': 4, 'dir': 'high_is_better'}
        }

    }

    # Define schema for the HTML renderer
    def get_footer(self, asset_type, configs=None):
        if not self.assets:
            return []

        # Get the column structure from the first asset
        asset_schema = self.assets[0].get_schema()
        
        if configs is None:
            configs = {m: self.get_metric_config(m, asset_type) for m in ['div_yield', 'div_growth', 'syield', 'iyield']}

        # Define which IDs in the schema should have footer values
        if asset_type == 'stocks':
            footer_map = {
                'ticker': {'label': 'Total Stocks', 'class': 'font-bold'},
                'price_eur': {'val': self.total_cost_basis, 'id': 'total-cost-basis-stocks', 'type': 'finance'},
                'market_value_eur': {'val': self.total_market_value, 'id': 'total-market-value-stocks', 'type': 'finance', 'suffix':'€'},
                'div_yield': {'val': configs['div_yield']['val'], 'id': 'portfolio-div-yield', 'type': 'monitor', 'suffix': '%',
                                    'bg_class': configs['div_yield']['class']},
                'div_growth': {'val': configs['div_growth']['val'], 'id': 'portfolio-div-growth', 'type': 'monitor', 'suffix': '%',
                                    'bg_class': configs['div_growth']['class']},
                'months_paid': {'type': 'visualizer'},
                'annual_dividend': {'val': self.annual_dividends, 'id': 'annual-dividends', 'type': 'finance', 'suffix': ' €'},
            }
        elif asset_type == 'crypto':
            footer_map = {
                'ticker': {'label': 'Total Crypto', 'class': 'font-bold'},
                'price_eur': {'val': self.total_cost_basis, 'id': 'total-cost-basis-crypto', 'type': 'finance', 'suffix':'€'},
                'market_value_eur': {'val': self.total_market_value, 'id': 'total-market-value-crypto', 'type': 'finance', 'suffix':'€'},
                'syield':{'val': self.portfolio_syield, 'id': 'portfolio_syield', 'type': 'monitor', 'suffix': '%', 'bg_class': configs['syield']['class']},
            }
        elif asset_type == 'interest':
            footer_map = {
                'currency': {'label': 'Total Interest', 'class': 'font-bold'},
                'market_value_eur': {'val': self.total_market_value, 'id': 'total_market_value-interest', 'type': 'finance', 'suffix':'€'},
                'iyield':{'val': self.portfolio_iyield, 'id': 'portfolio_iyield', 'type': 'monitor', 'suffix': '%', 'bg_class': configs['iyield']['class']},
            }

        resolved_footer = []
        current_gap = 0

        # Iterate through the Asset schema and find matches
        for col in asset_schema:
            if col['id'] in footer_map:
                # If there's a gap, add a spacer cell
                if current_gap > 0:
                    resolved_footer.append({'type': 'spacer', 'colspan': current_gap})
                
                # Add the data
                cell_data = footer_map[col['id']]
                cell_data['colspan'] = 1
                resolved_footer.append(cell_data)
                
                # Reset gap
                current_gap = 0
            else:
                # No footer value for this column, increment gap
                current_gap += 1

        # Add trailing spacer if needed
        if current_gap > 0:
            resolved_footer.append({'type': 'spacer', 'colspan': current_gap})

        return resolved_footer
        
    def get_metric_config(self, metric_name, asset_type):
        """Wrapper to call get_metric_config."""
        return get_metric_config(self, metric_name, self.METRIC_COLOUR_RULES, asset_type)
    

    def to_dict(self, asset_type):
        metric_configs = {
            m: self.get_metric_config(m, asset_type)
            for m in self.METRIC_COLOUR_RULES.get(asset_type, {}).keys()
        }
        # Define footer_data to update metrics in JS
        return {
            'assets': [a.to_dict() for a in self.assets],
            'total_market_value': self.total_market_value,
            'total_cost_basis': self.total_cost_basis,
            'monthly_income_data': vars(self.monthly_income_data), # vars is used to extract dicts from object instance
            'annual_dividends': self.annual_dividends,
            'portfolio_yield_data': vars(self.portfolio_yield_data),
            'portfolio_syield': self.portfolio_syield,
            'portfolio_iyield': self.portfolio_iyield,
            'sectors': vars(self.sectors),
            'status_colors': {m: cfg['class'] for m, cfg in metric_configs.items()},
            'footer': self.get_footer(asset_type, metric_configs)
        }

    def __repr__(self):
        return f"Portfolio(Assets={len(self.assets)}, TotalValue=€{self.total_market_value:.2f})"

# Load external data into a portfolio
class PortfolioLoader:
    @staticmethod
    def load_asset_data(asset_type, portfolio_store: PortfolioDataManager):
        manager = AssetDataManager(asset_type)
        #portfolio_mgr = PortfolioDataManager()
        
        asset_data = {
            'shares': manager.get_data('shares'),
            'price':  manager.get_data('price'),
            'env':    manager.get_data('env'),
            'soc':    manager.get_data('soc'),
            'gov':    manager.get_data('gov'),
            'cont':   manager.get_data('cont'),
            'syield': manager.get_data('syield'),
            'iyield': manager.get_data('iyield'),
            'fx_rates': portfolio_store.get_forex_rates(),
            'currency_choice': manager.get_data('currency'),
            'compounding': manager.get_data('compounding')
        }
        return asset_data


# Define the complete portfolio
class PortfolioManager:
    # Asset types whose Asset objects need FinanceDataManager metrics.
    # Everything else (e.g. 'interest') just needs ticker + user inputs.
    METRIC_ASSET_TYPES = {'stocks', 'crypto'}
    PROJECTED_INCOME_ASSET_TYPES = {'crypto', 'interest'}

    def __init__(self, portfolios_dict, free_cash=0.0, silent=False):
        self._portfolios = portfolios_dict
        self.free_cash = free_cash
        # Set attributes based on asset classes (stocks, crypto, etc.)
        for name, portfolio_obj in self._portfolios.items():
            setattr(self, name, portfolio_obj)
            
        # Show summary
        if not silent:
            logger.info(self)
            
    @property
    def total_market_value(self):
        return sum(p.total_market_value for p in self._portfolios.values())
            
    @property
    def grand_total_cost_basis(self):
        return sum(p.total_cost_basis for p in self._portfolios.values())
        
    @property
    def grand_total_with_cash(self):
        return self.total_market_value + self.free_cash
        
    # @property
    # def total_income_data(self):
    #     grand_total = [0.0] * 12
    #     grand_details = [[] for _ in range(12)]
        
    #     for portfolio in self.values():
    #         report = portfolio.monthly_income_data
    #         for i in range(12):
    #             grand_total[i] += report.payouts[i]
    #             if report.payouts[i] > 0:
    #                 # Add a header for the asset class in the hover text
    #                 grand_details[i].append(report.details[i])
                    
    #     return {
    #         "payouts": grand_total,
    #         "details": ["<br>".join(d) for d in grand_details]
    #     }

    @property
    def total_income_data(self):
        breakdown = {}
        grand_details = [[] for _ in range(12)]
        for name, portfolio in self.items():
            report = portfolio.monthly_income_data
            breakdown[name] = report.payouts          # e.g. breakdown['stocks'], ['crypto'], ['interest']
            for i in range(12):
                if report.payouts[i] > 0:
                    grand_details[i].append(report.details[i])
        breakdown['details'] = ["<br>".join(d) for d in grand_details]
        return breakdown

    def __iter__(self):
        return iter(self._portfolios)
        
    def keys(self):
        return self._portfolios.keys()
        
    def values(self):
        return self._portfolios.values()
        
    def items(self):
        return self._portfolios.items()
        
    def to_dict(self):
        data = {name: p.to_dict(asset_type=name) for name, p in self.items()}
        data['summary'] = {
            'total_market_value': self.total_market_value,
            'grand_total_cost_basis': self.grand_total_cost_basis,
            'grand_total_with_cash': self.grand_total_with_cash,
            'free_cash': self.free_cash,
            'total_income_data': self.total_income_data
        }
        return data
            
    def __repr__(self):
        count = 60
        # Header
        lines = [
            "\n" + "="*count,
            "PORTFOLIO SUMMARY",
            "="*count
        ]
        
        # Add each sub-portfolio
        for name, p in self._portfolios.items():
            lines.append(f" • {name.upper():<8}: {p.__repr__()}")
            
        # Add Global Totals
        lines.append("-" * count)
        lines.append(f" CASH       : €{self.free_cash:,.2f}")
        lines.append(f" TOTAL MV   : €{self.total_market_value:,.2f}")
        lines.append(f" GRAND TOTAL: €{self.grand_total_with_cash:,.2f}")
        lines.append("="*count + "\n")
        
        return "\n".join(lines)
    
    @classmethod
    @timed
    def from_storage(cls, asset_classes, finance_managers, portfolio_store: PortfolioDataManager, interval='1d', force_update=False):
        # Build a PortfolioManager
        portfolios = {}
        next(iter(finance_managers.values()))._ensure_fx_rates()   # ensure fx cache is fresh before building any asset
        
        for asset_type in asset_classes:
            dm = AssetDataManager(asset_type)
            tickers = dm.tickers

            if asset_type in cls.METRIC_ASSET_TYPES:
                fm = finance_managers[asset_type]
                raw_metrics = fm.get_metrics(tickers, interval=interval, force=force_update)
            else:
                raw_metrics = [{'Ticker': t, 'Value': 0} for t in tickers]

            # Load user-specific data
            data = PortfolioLoader.load_asset_data(asset_type, portfolio_store)
            
            # Create Asset objects
            assets = [
                Asset(
                    ticker=t,
                    metrics=next(m for m in raw_metrics if m['Ticker']==t),
                    asset_type=asset_type,
                    shares=data['shares'].get(t, 0),
                    price=data['price'].get(t, 0),
                    env=data['env'].get(t, 0),
                    soc=data['soc'].get(t, 0),
                    gov=data['gov'].get(t, 0),
                    cont=data['cont'].get(t, 0),
                    syield=data['syield'].get(t, 0),
                    iyield=data['iyield'].get(t, 0),
                    fx_rates=data['fx_rates'],
                    currency_choice=data['currency_choice'].get(t, ''),
                    compounding=data['compounding'].get(t, 'daily'),
                ) for t in tickers
            ]
            
            portfolios[asset_type] = Portfolio(assets)
            
        free_cash = portfolio_store.get_cash()
        return cls(portfolios, free_cash=free_cash)
    
    @classmethod
    def from_cache(cls, asset_classes, finance_managers, portfolio_store: PortfolioDataManager):
        """
        Rebuild portfolio from already-cached metrics. 
        No network calls, no staleness checks.
        """
        portfolios = {}
        next(iter(finance_managers.values()))._ensure_fx_rates()   # ensure fx cache is fresh before building any asset

        # Initialise portfolios with all asset types, even if empty
        for asset_type in asset_classes:
            portfolios[asset_type] = None  # Placeholder
            
        for asset_type in asset_classes:
            try:
                dm = AssetDataManager(asset_type)
                tickers = dm.tickers

                if asset_type in cls.METRIC_ASSET_TYPES:
                    if asset_type not in finance_managers:
                        logger.warning(f"No finance manager found for '{asset_type}'. Skipping.")
                        continue
                
                    fm = finance_managers[asset_type]
                    
                    # Read directly from in-memory cache, fall back to disk JSON
                    if not fm._static_metrics:
                        fm._static_metrics = fm._load_json(fm._metrics_path, default={})
                    raw_metrics = [fm._static_metrics[t] for t in tickers if t in fm._static_metrics]
                    valid_tickers = [t for t in tickers if t in fm._static_metrics]
                else:
                    raw_metrics = [{'Ticker': t, 'Value': 0} for t in tickers]
                    valid_tickers = tickers
                
                data = PortfolioLoader.load_asset_data(asset_type, portfolio_store)
                assets = [
                    Asset(
                        ticker=t,
                        metrics=next(m for m in raw_metrics if m['Ticker'] == t),
                        asset_type=asset_type,
                        shares=data['shares'].get(t, 0),
                        price=data['price'].get(t, 0),
                        env=data['env'].get(t, 0),
                        soc=data['soc'].get(t, 0),
                        gov=data['gov'].get(t, 0),
                        cont=data['cont'].get(t, 0),
                        syield=data['syield'].get(t, 0),
                        iyield=data['iyield'].get(t, 0),
                        fx_rates=data['fx_rates'],
                        currency_choice=data['currency_choice'].get(t, ''),
                        compounding=data['compounding'].get(t, 'daily'),
                    ) for t in valid_tickers
                ]
                portfolios[asset_type] = Portfolio(assets)
            except Exception as e:
                logger.error(f"Error processing asset type '{asset_type}': {e}")
                portfolios[asset_type] = Portfolio([])  # Ensure empty Portfolio is set

        free_cash = portfolio_store.get_cash()
        return cls(portfolios, free_cash=free_cash)


# Determine background colour. TODO: give user choice instead of hard coding
def get_metric_config(self, name, rule, asset_type):

    rules = rule.get(asset_type, {}).get(name.lower())

    if not rules:
        return {'val': 0, 'class': ''}
    
    # Logic to find the value on Asset vs Portfolio
    portfolio_attr = f"portfolio_{name}"
    if hasattr(self, name):
        val = getattr(self, name, 0)
    elif hasattr(self, portfolio_attr):
        val = getattr(self, portfolio_attr, 0)
    elif hasattr(self, 'portfolio_yield_data'):
        # Look inside the yield data namespace for Portfolio
        val = getattr(self.portfolio_yield_data, name, 0)
    else:
        val = 0

    # Resolve string thresholds
    low = rules['low']
    if isinstance(low, str):
        low = getattr(self, low, 0)
        
    high = rules['high']
    if isinstance(high, str):
        high = getattr(self, high, 0)

    return {
        'val': val,
        'class': calculate_color(val, low, high, rules['dir'])
    }

# Generic background colour function
def calculate_color(value, low, high, direction='high_is_better'):
    """Python implementation of your Jinja color logic."""
    if not isinstance(value, (int, float)) or value < 0:
        return "bg-red"
    
    if direction == 'high_is_better':
        if value < low: return "bg-red"
        if value <= high: return "bg-orange"
        return "bg-green"
    else: # low_is_better
        if value > high: return "bg-red"
        if value >= low: return "bg-orange"
        return "bg-green"
    