import json
import os
from flask import current_app
import tempfile
import logging
logger = logging.getLogger(__name__)

class JSONPersistenceManager:
    """Base class to handle atomic JSON operations."""
    
    def __init__(self, data_dir=None):
        self._data_dir = data_dir or current_app.config['DATA_FOLDER']
        os.makedirs(self._data_dir, exist_ok=True)

    def _get_path(self, filename):
        return os.path.join(self._data_dir, filename)

    def load(self, filename, default=None):
        path = self._get_path(filename)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return default if default is not None else {}
        
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error reading {path}: {e}")
            return default if default is not None else {}

    def save(self, filename, data):
        path = self._get_path(filename)
        dir_name = os.path.dirname(path)
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, suffix='.tmp') as tmp:
                json.dump(data, tmp, indent=4)
                tmp_path = tmp.name
            os.replace(tmp_path, path)
        except Exception as e:
            logger.error(f"Error saving to {path}: {e}")
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

class AssetDataManager(JSONPersistenceManager):
    """Manages data for a specific asset type (e.g., 'crypto', 'stocks')."""
    
    def __init__(self, asset_type):
        super().__init__()
        self.asset_type = asset_type
        self.filename = f"{asset_type}_data.json"
        self.data = self.load(self.filename, default={
            'tickers': [],
            'shares': {}, 'price': {}, 'env': {}, 
            'soc': {}, 'gov': {}, 'cont': {}, 'syield': {}, 'iyield': {},
            'fx_rates': {}, 'currency_choice':{}, 'compounding':{}
        })

    @property
    def tickers(self):
        return self.data.get('tickers', [])
    
    @tickers.setter
    def tickers(self, ticker_list):
        self.data['tickers'] = sorted(list(set(ticker_list)))
        self.save(self.filename, self.data)
    
    def get_data(self, kind):
        """Returns the specific dictionary (e.g., 'shares') from memory."""
        return self.data.get(kind, {})

    def save_data(self, kind, data):
        """Updates the in-memory dictionary and persists the whole file."""
        self.data[kind] = data
        self.save(self.filename, self.data)

    def update_ticker_metric(self, ticker, field, value):
        """Updates a specific metric for a ticker and saves."""
        if field not in self.data:
            self.data[field] = {}
        self.data[field][ticker] = value
        self.save(self.filename, self.data)

    def delete_ticker_globally(self, ticker):
        """Removes ticker from the list and all metric dictionaries."""
        # Remove from ticker list
        if ticker in self.data['tickers']:
            self.data['tickers'].remove(ticker)
        
        # Remove from all metric categories
        for key, value in self.data.items():
            if isinstance(value, dict):
                value.pop(ticker, None)
        
        self.save(self.filename, self.data)

    def update_ticker_metric_batch(self, kind, data_dict):
        self.data[kind] = data_dict
        self.save(self.filename, self.data)

class PortfolioDataManager(JSONPersistenceManager):
    """Handles global/cross-asset data like cash and forex."""
    
    def get_cash(self):
        return self.load('global_cash.json').get('free_cash', 0.0)

    def save_cash(self, amount):
        self.save('global_cash.json', {'free_cash': float(amount)})

    def get_forex_rates(self):
        data = self.load('last_fetch_date.json')
        return {k: v for k, v in data.items() if k.endswith('_rate')}
    
class AlertNotificationManager(JSONPersistenceManager):
    """Handles global/cross-asset persistent historical alerts using atomic JSON writes."""
    
    def __init__(self, data_dir=None):
        super().__init__(data_dir=data_dir)
        self.filename = 'alerts.json'

    def get_alerts(self):
        """Loads and returns the current list of alerts."""
        return self.load(self.filename, default=[])

    def save_alerts(self, alerts_list):
        """
        Persists alerts atomically while generating explicit logs tracking
        what changed before and after the operational write.
        """
        import inspect
        from flask import request

        # 1. Read what is securely saved on disk right now
        disk_alerts = self.get_alerts()
        len_before = len(disk_alerts)
        len_incoming = len(alerts_list)

        # Extract who called this Python method in the backend callstack
        caller_func = inspect.stack()[1].function

        logger.info(f"============================================================")
        logger.info(f"[ALERTS AUDIT] Route hit: {request.method} {request.path} (Endpoint: {request.endpoint})")
        logger.info(f"[ALERTS AUDIT] Triggered internally by Python function: '{caller_func}'")
        logger.info(f"[ALERTS AUDIT] Items on Disk BEFORE: {len_before} | Items INCOMING from UI: {len_incoming}")

        # 2. Check for suspicious empty overwrites
        if len_before > 0 and len_incoming == 0:
            logger.error(f"⚠️ [ALERTS AUDIT WARNING] DETECTED AN OVERWRITE! File had {len_before} alerts, but incoming payload is EMPTY.")
            
            # GUARDRAIL: If this isn't an explicit client POST hitting the alert manager route, abort the wipe!
            if not (request.endpoint == 'api.alerts' and request.method == 'POST'):
                logger.error(f"❌ [ALERTS AUDIT] Aborting destructive save. Preserving active disk data.")
                return

        # 3. Process safe merging logic
        disk_map = {a['id']: a for a in disk_alerts}
        for alert in alerts_list:
            alert_id = alert.get('id')
            if alert_id in disk_map:
                # Update status tags (e.g., read vs unread)
                disk_map[alert_id]['status'] = alert.get('status', disk_map[alert_id]['status'])
            else:
                # If it's a completely new alert, insert it at the front
                disk_alerts.insert(0, alert)

        # Handle genuine clear action explicitly triggered by user clicking "Clear All"
        if len_incoming == 0 and request.endpoint == 'api.alerts' and request.method == 'POST':
            disk_alerts = []

        # 4. Save to disk and evaluate what is there AFTER writing
        self.save(self.filename, disk_alerts)
        
        disk_alerts_after = self.get_alerts()
        logger.info(f"[ALERTS AUDIT] Items on Disk AFTER operational save: {len(disk_alerts_after)}")
        logger.info(f"============================================================")
        """
        Persists alerts atomically while generating explicit logs tracking
        what changed before and after the operational write.
        """
        import inspect
        from flask import request

        # 1. Look up what currently exists on disk BEFORE writing
        disk_alerts_before = self.get_alerts()
        len_before = len(disk_alerts_before)
        len_incoming = len(alerts_list)

        # Extract who called this Python method in the backend callstack
        caller_func = inspect.stack()[1].function

        logger.info(f"============================================================")
        logger.info(f"[ALERTS AUDIT] Route hit: {request.method} {request.path} (Endpoint: {request.endpoint})")
        logger.info(f"[ALERTS AUDIT] Triggered internally by Python function: '{caller_func}'")
        logger.info(f"[ALERTS AUDIT] Items on Disk BEFORE: {len_before} | Items INCOMING from UI: {len_incoming}")

        # 2. Check for suspicious empty overwrites
        if len_before > 0 and len_incoming == 0:
            logger.error(f"⚠️ [ALERTS AUDIT WARNING] DETECTED AN OVERWRITE! File had {len_before} alerts, but incoming payload is EMPTY.")
            
            # GUARDRAIL: If this isn't an explicit client POST hitting the alert manager route, abort the wipe!
            if not (request.endpoint == 'api.alerts' and request.method == 'POST'):
                logger.error(f"❌ [ALERTS AUDIT] Aborting destructive save. Preserving active disk data.")
                return

        # 3. Process safe merging logic using the correct variable name
        disk_map = {a['id']: a for a in disk_alerts_before}
        for alert in alerts_list:
            alert_id = alert.get('id')
            if alert_id in disk_map:
                # Update status tags (e.g., read vs unread)
                disk_map[alert_id]['status'] = alert.get('status', disk_map[alert_id]['status'])
            else:
                # If it's a completely new alert, insert it at the front
                disk_alerts_before.insert(0, alert)

        # Handle genuine clear action explicitly triggered by user clicking "Clear All"
        if len_incoming == 0 and request.endpoint == 'api.alerts' and request.method == 'POST':
            disk_alerts_before = []

        # 4. Save to disk and evaluate what is there AFTER writing
        self.save(self.filename, disk_alerts_before)
        
        disk_alerts_after = self.get_alerts()
        logger.info(f"[ALERTS AUDIT] Items on Disk AFTER operational save: {len(disk_alerts_after)}")
        logger.info(f"============================================================")
        """
        Persists alerts atomically while generating explicit logs tracking
        what changed before and after the operational write.
        """
        import inspect
        from flask import request

        # 1. Look up what currently exists on disk BEFORE writing
        disk_alerts_before = self.get_alerts()
        len_before = len(disk_alerts_before)
        len_incoming = len(alerts_list)

        # Extract who called this Python method in the backend callstack
        caller_func = inspect.stack()[1].function

        logger.info(f"============================================================")
        logger.info(f"[ALERTS AUDIT] Route hit: {request.method} {request.path} (Endpoint: {request.endpoint})")
        logger.info(f"[ALERTS AUDIT] Triggered internally by Python function: '{caller_func}'")
        logger.info(f"[ALERTS AUDIT] Items on Disk BEFORE: {len_before} | Items INCOMING from UI: {len_incoming}")

        # 2. Let's flag structural mutations explicitly
        if len_before > 0 and len_incoming == 0:
            logger.error(f"⚠️ [ALERTS AUDIT WARNING] DETECTED AN OVERWRITE! File had {len_before} alerts, but incoming payload is EMPTY.")
            
            # GUARDRAIL: If this isn't an explicit client POST hitting the alert manager route, abort the wipe!
            if not (request.endpoint == 'api.alerts' and request.method == 'POST'):
                logger.error(f"❌ [ALERTS AUDIT] Aborting destructive save. Preserving active disk data.")
                return

        # 3. Process safe merging logic
        disk_map = {a['id']: a for a in disk_alerts_before}
        for alert in alerts_list:
            alert_id = alert.get('id')
            if alert_id in disk_map:
                disk_map[alert_id]['status'] = alert.get('status', disk_map[alert_id]['status'])
            else:
                disk_alerts_before.insert(0, alert)

        # Handle genuine clear action explicitly triggered by user interaction
        if len_incoming == 0 and request.endpoint == 'api.alerts' and request.method == 'POST':
            disk_alerts_before = []

        # 4. Save to disk and evaluate what is there AFTER writing
        self.save(self.filename, disk_alerts_before)
        
        disk_alerts_after = self.get_alerts()
        logger.info(f"[ALERTS AUDIT] Items on Disk AFTER operational save: {len(disk_alerts_after)}")
        logger.info(f"============================================================")
        """
        Persists an array of alerts atomically to data/alerts.json by safely
        merging incoming client items with existing items on disk.
        """
        import inspect
        from flask import request

        # Read what is saved on disk right now
        disk_alerts_before = self.get_alerts()
        len_before = len(disk_alerts_before)
        len_incoming = len(alerts_list)

        # Extract who called this Python method in the backend callstack
        caller_func = inspect.stack()[1].function

        logger.info(f"============================================================")
        logger.info(f"[ALERTS AUDIT] Route hit: {request.method} {request.path} (Endpoint: {request.endpoint})")
        logger.info(f"[ALERTS AUDIT] Triggered internally by Python function: '{caller_func}'")
        logger.info(f"[ALERTS AUDIT] Items on Disk BEFORE: {len_before} | Items INCOMING from UI: {len_incoming}")

        # Create a lookup map of existing IDs on disk
        disk_map = {a['id']: a for a in disk_alerts_before}

        # Process incoming alerts from the frontend
        for alert in alerts_list:
            alert_id = alert.get('id')
            
            if alert_id in disk_map:
                # Update mutable tracking fields (like changing 'status' from 'unread' to 'read')
                disk_map[alert_id]['status'] = alert.get('status', disk_map[alert_id]['status'])
            else:
                # If it's a completely brand new threshold crossover alert, insert it at the top
                disk_alerts.insert(0, alert)
                
        # Handle explicit user deletions
        if len(alerts_list) == 0:
            disk_alerts = []

        # Atomically persist the safely merged array
        self.save(self.filename, disk_alerts)