class FinAppBase {
    constructor() {
        if (this.constructor === FinAppBase) {
            throw new TypeError("Cannot construct FinAppBase instances directly");
        }
        this.notifManager = new NotificationManager(this);
        this.loader = document.getElementById('loader');
        this._globalSyncInterval = null;
        this._globalSyncTimeout = null;
    }

    async initBase() {
        await this.notifManager.init();
        this._startGlobalSync();
    }

    // Global background processor
    _startGlobalSync() {
        const intervalMs = Math.min(
            window.APP_CONFIG?.liveIntervalMs    || 900000,
            window.APP_CONFIG?.researchIntervalMs || 86400000
        );

        this._runGlobalSync();

        // Then align to period boundaries
        const now = Date.now();
        const msUntilNext = intervalMs - (now % intervalMs);
        this._globalSyncTimeout = setTimeout(() => {
            this._runGlobalSync();
            this._globalSyncInterval = setInterval(
                () => this._runGlobalSync(), intervalMs
            );
        }, msUntilNext);
    }

    async _runGlobalSync() {
        try {
            const data = await fetch('/api/background_check').then(r => r.json());

            if (data.alerts) {
                this.notifManager.setAlerts(data.alerts);
            }
            if (data.assets) {
                this.notifManager.monitorStatusChanges(data);
            }
            if (typeof this.onBackgroundUpdate === 'function') {
                this.onBackgroundUpdate(data);
            }
        } catch (err) {
            console.error("Global background sync failed:", err);
        }
    }

    destroy() {
        if (this._globalSyncTimeout)  clearTimeout(this._globalSyncTimeout);
        if (this._globalSyncInterval) clearInterval(this._globalSyncInterval);
    }

    // Fetch logic
    async apiRequest(url, options = {}) {
        if (this.loader) this.loader.classList.remove('hidden');
        try {
            const response = await fetch(url, options);
            if (!response.ok) throw new Error(`Server Error: ${response.status}`);
            return await response.json();
        } catch (err) {
            console.error(`API Error: ${err}`);
            throw err;
        } finally {
            if (this.loader) this.loader.classList.add('hidden');
        }
    }

    // Ensure consistent currency format everywhere
    currencyFormat(value, currencyCode = 'EUR') {
        const val = typeof value === 'string' 
            ? parseFloat(value.replace(/\s/g, '').replace(',', '.')) 
            : value;
        if (isNaN(val) || val === 0) return `0.00 ${currencyCode === 'EUR' ? '€' : currencyCode}`;
        
        // Handle tiny values (scientific notation)
        if (Math.abs(val) < 0.01) {
            const symbol = currencyCode === 'EUR' ? '€' : currencyCode;
            return val.toExponential(2) + " " + symbol;
        }

        // Built-in currency formatter
        return new Intl.NumberFormat('fr-FR', {
            style: 'currency',
            currency: currencyCode,
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(val).replace(',', '.'); 
    }

    // Unified Plotly react wrapper
    renderPlot(containerId, figData, extraLayout = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;
        const fig = typeof figData === 'string' ? JSON.parse(figData) : figData;
        Plotly.react(container, fig.data, { ...fig.layout, ...extraLayout });
    }

    // Helper to safely update text content
    setText(id, text) {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    }
}

class PortfolioController extends FinAppBase {
    constructor(initialData, interval) {
        super();
        this.portfolioUpdateController = null;
        this.data = initialData;
        this.refreshInterval = interval || 900000;
        this.uiManager = new PortfolioUIManager(this);
        this.chartManager = new ChartManager(this);
        this.tickerManager = new TickerManager(this);
        this.init();
    }

    async init() {
        this.setupGlobalListeners();
        
        // Manager listeners
        this.uiManager.init();
        this.tickerManager.init();
        await this.initBase();

        // UI render
        if (this.data) {
            ['stocks', 'crypto', 'interest'].forEach(assetType => {
                this.updateUI(this.data, assetType);
            });
        }
        window.activeApp = this;
    }

    // Called by FinAppBase._runGlobalSync on every tick
    onBackgroundUpdate(data) {
        if (data.portfolio) {
            this.updateUI(data, 'stocks');
        }
        if (data.last_sync && document.getElementById('last-sync-time')) {
            document.getElementById('last-sync-time').textContent =
                `Last sync: ${data.last_sync}`;
        }
    }

    updateUI(data, assetType, ticker = null) {
        if (!data || !data.portfolio) return;

        // Update tables and text
        this.uiManager.update(data, assetType);

        // Update charts
        const subPortfolio = data.portfolio[assetType];
        this.chartManager.update(data, subPortfolio, assetType);
    }

    setupGlobalListeners() {
        // Event delegation for all portfolio inputs
        document.addEventListener('change', (e) => {
            if (e.target.matches('input[id*="_"],select[id*="_"]')) {
                this.handleInputChange(e.target);
            }
            if (e.target.id === 'free-cash-input') {
                this.saveCashValue(e.target.value);
            }
        });
    }

    async handleInputChange(input) {
        const ticker = input.id.split('_').pop();
        const section = input.closest('[id$="-section"]');
        const assetType = section ? section.id.split('-')[0] : 'stocks';

        // Persistent save
        await this.saveSingleValue(input, ticker, input.value);
            // .then(data => {
            //     if (data.portfolio) {
            //         PortfolioUI.update(data, assetType, ticker);
            //     }
            // });
    }

    getTickersFromPage() {
        //console.log("getTickersFromPage called");
        const tickerDiv = document.getElementById('ticker-list');
        if (tickerDiv && tickerDiv.dataset.tickers) {
            return tickerDiv.dataset.tickers.split(',');
        }
        return [];
    }
        
    // Prevent form submission on Enter keypress
    handleEnterKey(e) {
        if (e.key === 'Enter') {
            // Stop the browser's default action (submitting the form)
            e.preventDefault(); 
            // Remove focus from the input field after pressing Enter
            e.target.blur();
        }
    }

    // Collect all necessary data from the table
    collectTableData(assetType) {
        let assets = [];
        
        //console.log("collectTableData called");
        // Find hidden inputs named "tickers" in the table
        const tickerInputs = document.querySelectorAll('input[name="tickers"]');

        // Find all rows (tr) in the table body
        tickerInputs.forEach(input => {
            const ticker = input.value;
            // Retrieve the ticker symbol, assuming it's stored as a data attribute on the row or an element inside.
            // Adjust the selector if input fields are not inside a <tr>
            const sharesInput = document.querySelector(`input[name="shares_${ticker}"]`);
            const priceInput = document.querySelector(`input[name="price_${ticker}"]`);
            const envInput = document.querySelector(`input[name="env_${ticker}"]`);
            const socInput = document.querySelector(`input[name="soc_${ticker}"]`);
            const govInput = document.querySelector(`input[name="gov_${ticker}"]`);
            const contInput = document.querySelector(`input[name="cont_${ticker}"]`);
            const syieldInput = document.querySelector(`input[name="syield_${ticker}"]`);
            const iyieldInput = document.querySelector(`input[name="iyield_${ticker}"]`);

            if (sharesInput) {
                // Default values to 0
                let envVal = 0, socVal = 0, govVal = 0, contVal= 0, syieldVal=0, iyieldVal=0;
                
                // Only attempt to read ESG values if assetType is 'stocks'
                if (assetType === 'stocks') {
                    envVal = envInput ? (parseInt(envInput.value) || 0) : 0;
                    socVal = socInput ? (parseInt(socInput.value) || 0) : 0;
                    govVal = govInput ? (parseInt(govInput.value) || 0) : 0;
                    contVal = contInput ? (parseInt(contInput.value) || 0) : 0;
                }

                // Only attempt to read staking yields if assetType is 'crypto'
                if (assetType === 'crypto') {
                    syieldVal = syieldInput ? (parseFloat(syieldInput.value) || 0) : 0;
                }

                // Only attempt to read interest yields if assetType is 'interest'
                if (assetType === 'interest') {
                    iyieldVal = iyieldInput ? (parseFloat(iyieldInput.value) || 0) : 0;
                }
        
                assets.push({
                    ticker: ticker,
                    shares: sharesInput ? (parseFloat(sharesInput.value) || 0.0) : 0.0,
                    price: priceInput ? (parseFloat(priceInput.value) || 0.0) : 0.0,
                    // Check if ESG inputs exist before accessing .value
                    env: envVal,
                    soc: socVal,
                    gov: govVal,
                    cont: contVal,
                    syield: syieldInput ? (parseFloat(syieldInput.value) || 0.0) : 0.0,
                    iyield: iyieldInput ? (parseFloat(iyieldInput.value) || 0.0) : 0.0,
                });
            }
        });
        
        return { assets: assets };
    }
    
    // Send data to Flask and update the plot
    async updatePortfolioUI(assetType, ticker = null) {
        console.log(`updateUI called for ${assetType}`);

        // If there's an ongoing request, cancel it
        if (this.portfolioUpdateController) {
            this.portfolioUpdateController.abort();
            console.log("Previous request aborted to prioritise new input.");
        }

        this.portfolioUpdateController = new AbortController();
        const dataToSend = this.collectTableData(assetType);
        
        try {
            const data = await this.apiRequest(`/update_portfolio_data/${assetType}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(dataToSend),
                signal: this.portfolioUpdateController.signal
            });

            this.updateUI(data, assetType, ticker);
            this.portfolioUpdateController = null;
        } catch (error) {
            if (error.name !== 'AbortError') {
                console.error(`Error updating ${assetType} portfolio:`, error);
            }
        }
    }
    
    // Send a single updated value to the server
    async saveSingleValue(inputElement, ticker, value) {
        console.log("saveSingleValue called");
        const fieldType = inputElement.id.split('_')[0]; // Remove underscore to get the field
        
        // Determine the asset type by looking at the parent container
        const assetType = inputElement.closest('[id$="-section"]').id.split('-')[0]; // Return 'stocks', 'crypto', etc...

        const isSelect = inputElement.tagName === 'SELECT';

        try {
            const data = await this.apiRequest(`/save_single_value/${assetType}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    ticker: ticker,
                    field: fieldType,
                    value: isSelect ? value : (parseFloat(value) || 0.0),
                    asset_type: assetType
                })
            });
            
            if (data && data.portfolio) {
                this.updateUI(data, assetType, ticker);
            }
        } catch (error) {
            alert("Failed to save value.");
        }
    }
    
    async saveCashValue(value) {
        try {
            const data = await this.apiRequest('/save_cash', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cash: value })
            });
            
            // Re-fetch full data to update grand totals and allocation charts
            const freshData = await this.apiRequest('/update_portfolio_data', { method: 'POST' });
            this.updateUI(freshData, 'stocks');
        } catch (error) {
            console.error('Error saving cash:', error);
        }
    }
}

class ChartManager {
    /**
     * @param {PortfolioController} app - Reference to the main controller
     */
    constructor(app) {
        this.app = app;
        // Configuration for the marker colors to keep them consistent
        this.colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6'];
    }

    /**
     * Main entry point called by the controller
     */
    update(data, subPortfolio, assetType) {
        console.log(`ChartManager: Updating charts for ${assetType}`);
        
        // Update Asset Allocation and Sector Diversification
        this.refreshPieCharts(data, subPortfolio, assetType);

        // Update the Monthly Income Bar Chart (specific to the portfolio view)
        const plotDiv = document.getElementById('monthly-income-plot');
        if (plotDiv && data.income_plot) {
            this.app.renderPlot('monthly-income-plot', data.income_plot);
        }
    }

    refreshPieCharts(data, subPortfolio, assetType) {
        console.log(`refreshPieCharts called for asset type ${assetType}`);
        const manager = data.portfolio;
        
        // Prepare data for Global Allocation
        const allTotals = {
            stocks: manager.stocks.total_market_value || 0,
            crypto: manager.crypto.total_market_value || 0,
            interest: manager.interest.total_market_value || 0,
            cash: manager.summary.free_cash || 0
        };

        const categories = ['stocks','crypto','interest','cash']; //TODO add assets
        let allocLabels = [];
        let allocValues = [];

        categories.forEach(cat => {
            const val = allTotals[cat];
            if (val > 0) {
                allocLabels.push(cat.charAt(0).toUpperCase() + cat.slice(1));
                allocValues.push(val);
                console.log("category: ", cat);
                console.log("val: ", val);
            }
        });

        // Update Global Allocation Chart
        this.renderPieChart(
            'asset-allocation-chart', 
            allocLabels, 
            allocValues, 
            'Global allocation'
        );

        // Update Sector Diversification (Only if on Stocks tab and data exists)
        if (assetType === 'stocks' && subPortfolio.sectors.labels && subPortfolio.sectors) {
            this.renderPieChart(
                'sector-chart', 
                subPortfolio.sectors.labels, 
                subPortfolio.sectors.values, 
                'Sector diversification'
            );
        }
    }

    /**
     * Internal helper to build the Plotly Pie trace and layout
     */
    renderPieChart(elementId, labels, values, chartTitle) {
        if (!document.getElementById(elementId)) return;

        const total = values.reduce((sum, v) => sum + v, 0); // Calculate total for percentage calculation
        const customText = labels.map((_, i) => {
            const percent = ((values[i] / total) * 100).toFixed(2);
            return `${percent}%`;
        });

        const trace = [{
            values: values,
            labels: labels,
            type: 'pie',
            hole: 0.4,
            text: customText,
            textinfo: "label+text",
            textposition: "outside",
            automargin: false,
            domain: {
                x: [0.15, 0.85],
                y: [0.15, 0.85]  // Gives space at bottom and top (for title)
            },
            hovertemplate: "<b>%{label}</b><br>%{text}<br><extra></extra>",
            hoverlabel: {
                align: 'center'
            },
            marker: { colors: this.colors }
        }];

        const layout = {
            title: {
                text: chartTitle,
                font: { color: '#ffffff', size: 18 },
                y: 0.95
            },
            autosize: true,
            margin: { t: 40, b: 20, l: 20, r: 20 },
            showlegend: false,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ffffff' }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.react(elementId, trace, layout, config);
    }
}

class PortfolioUIManager {
    /**
     * @param {PortfolioController} app - Reference to the main controller
     */
    constructor(app) {
        this.app = app;
    }

    init() {
        this.setupListeners();
        this.hydrateInitialData();
    }

    setupListeners() {
        document.addEventListener('keydown', (e) => {
            // Matches ticker inputs (id contains _) OR the cash input
            if (e.target.matches('input[id*="_"]') || e.target.id === 'free-cash-input') {
                if (e.key === 'Enter') {
                    this.app.handleEnterKey(e); 
                }
            }
        });

        document.addEventListener('change', (e) => {
        });
    }
    
    // Sync JS with the data Jinja already put in the HTML
    hydrateInitialData() {
        // Pass the initial JSON from Python
        // if (window.INITIAL_PORTFOLIO_DATA) {
        //     this.update(window.INITIAL_PORTFOLIO_DATA, 'stocks');
        // }
        if (this.app.data) {
            this.update(this.app.data, 'stocks');
        }
    }
    
    // Main entry point called after the fetch request
    // update(data, assetType, ticker = null) {
    update(data, assetType) {
        if (!data || !data.portfolio) {
            console.error("Data structure invalid", data);
            //if (this.dom.loader) this.dom.loader.classList.add('hidden');
            return;
        }

        const manager = data.portfolio;
        const subPortfolio = manager[assetType];
        const summary = manager.summary;

        // Update the grand total
        this.app.setText('grand-total-display', this.app.currencyFormat(summary.grand_total_with_cash));

        // Update footer cells
        if (subPortfolio.footer) {
            subPortfolio.footer.forEach(cell => {
                // Only update if the cell has an ID and a value
                if (cell.id && cell.val !== undefined) {
                    let displayVal = cell.val;
                    
                    // Format based on type
                    if (cell.type === 'finance' || cell.type === 'monitor') {
                        // Use currency format or percentage based on suffix
                        displayVal = cell.suffix === '%' 
                            ? cell.val.toFixed(2) + '%' 
                            : this.app.currencyFormat(cell.val);
                    }
                    
                    this.app.setText(cell.id, displayVal);

                    if (cell.bg_class) {
                        this.setCellColor(cell.id, cell.bg_class);
                    }
                }
            });
        }

        // Update table metrics
        this.syncMetrics(subPortfolio);

        // Update income
        if (assetType === 'stocks' && subPortfolio.monthly_income_data) {
            const dataRow = document.getElementById('total-month-data-row');
            if (dataRow) {
                dataRow.innerHTML = subPortfolio.monthly_income_data.counts.map(count => `
                    <span class="month-data-item ${count > 0 ? 'paid' : ''}">${count}</span>
                `).join('');
            }
        }

        // Update all charts
        this.app.chartManager.update(data, subPortfolio, assetType);

        // Post-update logic
        //this.app.monitorStatusChanges();
        if (this.app.notifManager) {
            this.app.notifManager.monitorStatusChanges(data);
        }
    }

    setCellColor(id, bgClass) {
        const el = document.getElementById(id);
        if (!el) return;

        el.classList.remove("bg-green", "bg-orange", "bg-red");

        if (bgClass) {
            el.classList.add(bgClass);
        }
    }
    
    syncMetrics(subPortfolio) {
        subPortfolio.assets.forEach(asset => {
            // Loop through the schema defined in Python
            asset.schema.forEach(column => {
                const metric = column.id;
                const selector = `[id="${metric}_${asset.ticker}"], [name="${metric}_${asset.ticker}"]`;
                const elements = document.querySelectorAll(selector);

                elements.forEach(el => {
                    // Colours
                    const colorClass = asset.status_colors[metric];
                    const target = el.tagName === 'INPUT' ? el.closest('td') : el;
                    
                    if (target) {
                        target.classList.remove('bg-red', 'bg-orange', 'bg-green', 'bg-grey');
                        if (colorClass) target.classList.add(colorClass);
                    }

                    // Values
                    if (document.activeElement === el) return; // Don't interrupt typing

                    const rawVal = asset[metric];
                    if (rawVal === undefined || rawVal === null) return;

                    // Check the type from get_schema()
                    if (column.type === 'monitor_input' || column.type === 'select_input' || el.tagName === 'INPUT') {
                        // Inputs: update .value to preserve the box
                        el.value = rawVal; 
                    } else if (column.type !== 'visualizer' && column.type !== 'ticker') {
                        // Display cells: apply suffixes and formatting
                        const suffix = column.suffix || '';
                        let displayVal;

                        if (column.type === 'finance' && suffix === ' €') {
                            displayVal = this.app.currencyFormat(rawVal);
                        } else if (typeof rawVal === 'number') {
                            // Only use decimals for non-ESG numbers
                            displayVal = rawVal.toFixed(2) + suffix;
                        } else {
                            displayVal = rawVal + suffix;
                        }
                        el.textContent = displayVal;
                    }
                });
            });
        });
    }
}

class NotificationManager {
    constructor(app) {
        this.app = app;
        this.alerts = [];
        // Cache DOM elements
        this.dom = {
            bellBtn: document.getElementById('bellBtn'),
            bellBadge: document.getElementById('bell-badge'),
            list: document.getElementById('notification-list')
        };
    }

    async init() {
        if (!this.dom.bellBtn || !this.dom.list) return;
        this.setupListeners();
        // Load persistent historical alerts from server upon page mount
        await this.loadAlerts();
    }

    //////////////////////
    // Persistence methods
    async loadAlerts() {
        try {
            const data = await fetch('/api/alerts').then(res => res.json());
            this.setAlerts(data.alerts || []);
        } catch (err) {
            console.error("Failed to load persistent alerts:", err);
        }
    }

    // TODO
    async _saveToBackend() {
        // --- DIAGNOSTIC LOGGING ---
        console.warn(`[CLIENT SAVE] Pushing to /api/alerts. Total items: ${this.alerts.length}`);
        if (this.alerts.length === 0) {
            console.error("[CLIENT SAVE WARNING] Sending a blank array to the server! Trace:", new Error().stack);
        }
        // --------------------------
        try {
            await fetch('/api/alerts', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ alerts: this.alerts })
            });
        } catch (err) {
            console.error("Failed to persist alerts to server:", err);
        }
    }

    setAlerts(alertsArray) {
        this.alerts = alertsArray;
        this.render();
    }
    ////////////////////////

    ///////////////
    // UI rendering
    setupListeners() {
        this.dom.bellBtn.addEventListener('click', (e) => {
            e.stopPropagation(); // Prevent clicks from closing the menu immediately
            const list = this.dom.list;
            const isVisible = list.style.display === 'block';
            list.style.display = isVisible ? 'none' : 'block';
            if (!isVisible) this.markAllAsRead();
        });

        // Close on outside click
        document.addEventListener('click', () => {
            if (this.dom.list) this.dom.list.style.display = 'none';
        });

        // Prevent the list itself from closing when clicking inside it
        this.dom.list.addEventListener('click', (e) => e.stopPropagation());
    }

    render() {
        const unreadCount = this.alerts.filter(a => a.status === 'unread').length;
        const appRef = 'window.activeApp';
        // Update badge
        if (this.dom.bellBadge) {
            this.dom.bellBadge.style.display = unreadCount > 0 ? 'block' : 'none';
            this.dom.bellBadge.textContent = unreadCount;
        }

        // Update list content
        let itemsHTML = this.alerts.map(a => `
            <li onclick="(${appRef}).notifManager.markAsRead('${a.id}')"
                style="padding: 12px; border-bottom: 1px solid #eee; background: ${a.status === 'unread' ? '#f0f7ff' : 'white'}; cursor: pointer; transition: background 0.2s;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
                    <small style="color: #888; font-size: 0.8em;">${a.time}</small>
                    ${a.status === 'unread'
                        ? '<span style="width:8px; height:8px; background:#007BFF; border-radius:50%; margin-top: 4px;"></span>'
                        : ''}
                </div>
                <div style="font-size: 0.9em; color: #333; line-height: 1.3;">
                    <strong style="color: #111;">${a.title}</strong>: ${a.message}
                </div>
            </li>
        `).join('');
        
        // Wrap list items inside a constrained rolling height container
        const scrollContainerHTML = itemsHTML 
            ? `<ul style="list-style: none; margin: 0; padding: 0; max-height: 280px; overflow-y: auto;">
                   ${itemsHTML}
               </ul>`
            : '<div style="padding: 20px 10px; text-align: center; color: #666; font-size: 0.9em;">No notifications</div>';
        
        // Add "Clear All" layout component button
        const clearBtnHTML = this.alerts.length > 0 ? `
            <div style="padding: 10px; text-align: center; border-top: 1px solid #ddd; background: #f9f9f9;">
                <button onclick="(${appRef}).notifManager.clearAll()"
                        style="cursor: pointer; border: 1px solid #ccc; background: white;
                               padding: 5px 12px; border-radius: 4px; font-size: 0.85em; font-weight: 500;
                               color: #555; hover: background: #f0f0f0;">
                    Clear All
                </button>
            </div>` : '';

        // Composite full component string into the DOM target layout
        this.dom.list.innerHTML = scrollContainerHTML + clearBtnHTML;
    }
    /////////////////////

    ////////////////////
    // Alert logic
    async add(message, type = 'info', title = 'Alert') {
        this.alerts.unshift({
            id: Date.now() + Math.random().toString(36).substr(2, 5), // Unique ID tracking
            time: new Date().toLocaleTimeString(),
            title: title,
            message: message,
            type: type,
            status: 'unread'
        });
        this.render();
        
        // Persist to alerts.json instantly
        await this._saveToBackend();
    }

    // Handles portfolio status transition checks
    addStatusAlert(ticker, metric, oldStatus, newStatus) {
        // Transform codes to user-friendly text labels
        const labels = { 'good': 'good', 'caution': 'caution', 'bad': 'bad' };
        const colors = { 'good': '#28a745', 'caution': '#fd7e14', 'bad': '#dc3545'};
        const oldSpan = `<span style="color: ${colors[oldStatus]}; font-weight: 600;">${labels[oldStatus]}</span>`;
        const newSpan = `<span style="color: ${colors[newStatus]}; font-weight: 600;">${labels[newStatus]}</span>`;
        this.add(
            `${ticker} ${metric.replace(/_/g, ' ')} changed from ${oldSpan} to ${newSpan}.`,
            'status_change',
            'Portfolio threshold cross'
        );
    }

    monitorStatusChanges(newData) {
        // Expects the raw portfolio structure from background sync
        if (!newData?.assets) return;

        console.log("Monitor received structure:", localStorage.getItem('portfolio_state'));
        //const previousState = JSON.parse(localStorage.getItem('portfolio_state') || "{}");
        //let stateChanged = false;

        Object.entries(newData.assets).forEach(([ticker, assetData]) => {
            if (!assetData.metrics) return;

            Object.entries(assetData.metrics).forEach(([metricName, metricData]) => {
                // Check if backend supplied direct delta statuses
                if (metricData.old_status && metricData.new_status) {
                    console.log(`Color change detected for ${ticker} ${metricName}: ${metricData.old_status} -> ${metricData.new_status}`);
                    
                    // Directly trigger the alert banner
                    this.addStatusAlert(ticker, metricName, metricData.old_status, metricData.new_status);
                } else {
                    // FALLBACK: Keeps your automatic background sync syncs working as normal
                    // using standard metric validation loops if coming from /api/background_check
                    const previousState = JSON.parse(localStorage.getItem('portfolio_state') || "{}");
                    const stateKey = `${ticker}_${metricName}`;
                    const { value: val, green_limit, red_limit } = metricData;
                    if (val === undefined || green_limit === undefined) return;

                    const currentStatus = val >= green_limit ? 'good' : val <= red_limit ? 'bad' : 'caution';
                    const oldStatus = previousState[stateKey];

                    if (oldStatus !== undefined && oldStatus !== currentStatus) {
                        this.addStatusAlert(ticker, metricName, oldStatus, currentStatus);
                    }
                    previousState[stateKey] = currentStatus;
                    localStorage.setItem('portfolio_state', JSON.stringify(previousState));
                }
            });
        });
    }

    // Triggered automatically whenever the user clicks the bell icon to open the window
    markAllAsRead() {
        const hasUnread = this.alerts.some(a => a.status === 'unread');
        if (!hasUnread) return;

        this.alerts.forEach(a => a.status = 'read');
        this.render();
        this._saveToBackend();
    }

    // Triggered when clicking a specific notification item in the list
    markAsRead(alertId) {
        const alert = this.alerts.find(a => a.id === alertId);
        if (alert?.status === 'unread') {
            alert.status = 'read';
            this.render();
            this._saveToBackend();
        }
    }

    // Triggered by the "Clear All" button to wipe the list clean
    clearAll() {
        this.alerts = [];
        this.render();
        this._saveToBackend();
    }
        
}

class TickerManager {
    /**
     * @param {PortfolioController} app - Reference to the main controller
     */
    constructor(app) {
        this.app = app;
        
        // Cache DOM elements
        this.dom = {
            btnMain: document.getElementById('add-ticker-btn'),
            btnCancel: document.getElementById('cancel-ticker-btn'),
            form: document.getElementById('ticker-form'),
            selector: document.getElementById('category-selector'),
            label: document.getElementById('category-label'),
            input: document.getElementById('ticker'),
            tickerLabel: document.getElementById('ticker-label'),
            currencyField: document.getElementById('currency-field'),
            currencyInput: document.getElementById('currency'),
            submitBtn: document.getElementById('submit-button')
        };
    }

    init() {
        this.setupListeners();
    }

    setupListeners() {
        // Toggle the category picker
        if (this.dom.btnMain) {
            this.dom.btnMain.onclick = () => this.toggleCategoryPicker();
        }

        // Cancel and go back
        if (this.dom.btnCancel) {
            this.dom.btnCancel.onclick = () => this.resetUI();
        }

        // Handle category selection (e.g., clicking "Stocks")
        document.querySelectorAll('.category-selector').forEach(btn => {
            btn.onclick = (e) => {
                const cat = e.currentTarget.getAttribute('data-category');
                this.showForm(cat);
            };
        });

        // Add Ticker (Form Submission)
        if (this.dom.form) {
            this.dom.form.addEventListener('submit', (e) => this.handleAdd(e));
        }

        // Delete Ticker (Event Delegation)
        document.addEventListener('click', (e) => {
            const deleteBtn = e.target.closest('.delete-btn');
            if (deleteBtn) {
                e.preventDefault();
                e.stopPropagation();
                this.handleDelete(deleteBtn);
            }
        });
    }

    toggleCategoryPicker() {
        const isSelectorVisible = this.dom.selector.style.display === 'block';
        const isFormVisible = this.dom.form.style.display === 'block';

        if (isSelectorVisible || isFormVisible) {
            this.resetUI();
        } else {
            this.dom.selector.style.display = 'block';
            this.dom.form.style.display = 'none';
        }
    }

    showForm(category) {
        this.dom.selector.style.display = 'none';
        this.dom.form.style.display = 'block';
        // Set form action dynamically based on chosen category
        this.dom.form.action = `/add/${category}`;
        this.dom.label.textContent = category.charAt(0).toUpperCase() + category.slice(1);
        this.dom.input.focus();

        // For the interest asset class
        const isInterest = category === 'interest';

        // Symbol becomes optional "Name" for interest, currency field appears/becomes required
        if (this.dom.tickerLabel) {
            this.dom.tickerLabel.textContent = isInterest ? 'Name (optional):' : 'Symbol:';
        }
        this.dom.input.required = !isInterest;

        if (this.dom.currencyField) {
            this.dom.currencyField.style.display = isInterest ? 'block' : 'none';
        }
        if (this.dom.currencyInput) {
            this.dom.currencyInput.required = isInterest;
        }
    }

    resetUI() {
        this.dom.form.style.display = 'none';
        this.dom.selector.style.display = 'none';
        this.dom.form.reset();
        if (this.dom.currencyField) this.dom.currencyField.style.display = 'none';
        if (this.dom.tickerLabel) this.dom.tickerLabel.textContent = 'Symbol:';
        this.dom.input.required = true;
    }

    async handleAdd(e) {
        e.preventDefault();
        this.dom.submitBtn.textContent = 'Adding...';
        this.dom.submitBtn.disabled = true;

        
        try {
            // Use the app's apiRequest for consistent loading/error handling
            const formData = new FormData(this.dom.form);
            await this.app.apiRequest(this.dom.form.action, {
                method: 'POST',
                body: formData
            });

            // If add is successful, usually a reload is cleanest to fetch new prices
            window.location.reload();
        } catch (err) {
            console.error('Add ticker failed:', err);
            this.dom.submitBtn.textContent = 'Save Asset';
            this.dom.submitBtn.disabled = false;
            alert("Failed to add ticker. Check console.");
        }
    }

    async handleDelete(btn) {
        const tickerId = btn.getAttribute('data-id');
        if (!confirm(`Are you sure you want to remove ${tickerId}?`)) return;

        const section = btn.closest('[id$="-section"]');
        const assetType = section ? section.id.split('-')[0] : 'stocks';

        try {
            const data = await this.app.apiRequest(`/delete/${encodeURIComponent(assetType)}/${encodeURIComponent(tickerId)}`, {
                method: 'POST'
            });

            if (data.status === 'success') {
                // Visual feedback: fade out row
                const row = document.getElementById(`row-${tickerId}`);
                if (row) {
                    row.classList.add('row-fade-out');
                    setTimeout(() => row.remove(), 400);
                }

                // Tell the app to update the totals/charts with the new data from server
                if (data.portfolio) {
                    this.app.updateUI(data, assetType);
                }
            }
        } catch (error) {
            console.error('Delete failed:', error);
            alert('Could not delete ticker.');
        }
    }
}

// class SyncManager {
//     /**
//      * @param {PortfolioController} app - Reference to the main controller
//      */
//     constructor(controller) {
//         this.controller = controller;
//         this.dom = {
//             syncBtn: document.getElementById('sync-button'),
//             lastSyncText: document.getElementById('last-sync-time')
//         };
//         this.init();
//     }

//     init() {
//         if (this.dom.syncBtn) {
//             this.dom.syncBtn.addEventListener('click', () => this.sync(true));
//         }

//         // Link page-specific synchronization directly into the global app sync hook
//         this.controller.onBackgroundUpdate = (data) => {
//             console.log("Global sync tick detected on portfolio view.");
//             if (data.portfolio_updated && data.portfolio_data) {
//                 this.controller.ui.update(data.portfolio_data);
//                 if (this.dom.lastSyncText && data.last_sync) {
//                     this.dom.lastSyncText.textContent = `Last Sync: ${data.last_sync}`;
//                 }
//             }
//         };
//     }

//     async sync(force = false) {
//         console.log("SyncManager: syncing at", new Date().toLocaleTimeString());

//         try {
//             const data = await this.controller.apiRequest(`/sync?force=${force}`);
//             this.controller.ui.update(data);
//             if (this.dom.lastSyncText && data.last_sync) {
//                 this.dom.lastSyncText.textContent = `Last Sync: ${data.last_sync}`;
//             }
//         } catch (err) {
//             alert("Sync processing failed. Check runtime logs.");
//         }
//     }
// }

class ResearchController extends FinAppBase {
    constructor(config) {
        super();
        this.state = {
            currentTicker: config.selectedTicker,
            secondaryTicker: "",
            currentMode: "price",
            portfolioMode: "returns"
        };
        this.dom = {
            refreshBtn: document.getElementById('refresh-frontier-btn'),
            portfolioContainer: document.getElementById('portfolio-plot-container'),
            tickerContainer: document.getElementById('price-plot-container'),
            portfolioTitle: document.getElementById('portfolio-chart-title'),
            tickerTitle: document.getElementById('chart-title'),
            tickerRows: document.querySelectorAll('.ticker-row'),
            portfolioTabs: document.querySelectorAll('.portfolio-tab'),
            tickerTabs: document.querySelectorAll('.tab'),
            startDateInput: document.getElementById('start-date'),
            plotMessage: document.getElementById('plot-message'),
            expandBtn: document.getElementById('expand-btn')
        };
        this.init();
    }

    async init() {
        await this.initBase();
        this.registerEventListeners();
        this.updateView();
        this.updatePortfolioView();
        this.refreshSidebarUI();

        // Reflow plots when container resizes
        if (this.dom.tickerContainer) {
            const resizeObserver = new ResizeObserver(() => {
                // Check if Plotly has drawn inside the container before trying to relayout
                if (this.dom.tickerContainer.data) {
                    Plotly.relayout(this.dom.tickerContainer, {
                        height: this.dom.tickerContainer.offsetHeight
                    });
                }
            });
            resizeObserver.observe(this.dom.tickerContainer);
        }
        window.activeApp = this;
    }

    // Asset Level Logic
    async updateView() {
        if (!this.dom.tickerContainer) return;
        if (this.dom.tickerTitle) {
            this.dom.tickerTitle.innerText = this.state.currentTicker;
        }
        
        // Clear any previous message
        if (this.dom.plotMessage) {
            this.dom.plotMessage.style.display = 'none';
            this.dom.plotMessage.innerText = '';
        }

        let url = `/get_data?ticker=${this.state.currentTicker}&mode=${this.state.currentMode}`;
        if (this.state.currentMode === 'map-2dcorr' && this.state.secondaryTicker) {
            url += `&ticker2=${this.state.secondaryTicker}`;
        }

        try {
            const data = await this.apiRequest(url);
            if (data.error || data.warning) {
                // Show message without touching the Plotly container
                if (this.dom.plotMessage) {
                    this.dom.plotMessage.style.color = data.error ? 'red' : 'orange';
                    this.dom.plotMessage.innerText = data.error || data.warning;
                    this.dom.plotMessage.style.display = 'block';
                }
                // Clear the plot so stale data isn't shown
                Plotly.purge(this.dom.tickerContainer);
                return;
            }
            const plotData = JSON.parse(data.fig_data);
            await Plotly.react(
                this.dom.tickerContainer,
                plotData.data,
                { ...plotData.layout, autosize: true },
                data.config || {}
            );
            Plotly.relayout(this.dom.tickerContainer, {
                height: this.dom.tickerContainer.offsetHeight || 500
            });
            this.preventScrollOnDropdown('price-plot-container');
            this.updateMetrics(data.metrics);
        } catch (err) {
            console.error("Asset Plot Error:", err);
            this.dom.tickerContainer.innerHTML = '<p style="color:red;">Error loading chart.</p>';
        }
    }

    preventScrollOnDropdown(elementId) {
        const container = document.getElementById(elementId);
        if (!container) return;

        // Plotly renders dropdowns as .updatemenu-container
        const observer = new MutationObserver(() => {
            const dropdowns = container.querySelectorAll(
                '.updatemenu-container, .updatemenu-dropdown-button'
            );
            dropdowns.forEach(el => {
                el.addEventListener('wheel', (e) => {
                    e.stopPropagation();
                    e.preventDefault();
                }, { passive: false });
            });
        });

        observer.observe(container, { childList: true, subtree: true });
    }

    updateMetrics(metrics) {
        if (!metrics) return;
        Object.entries(metrics).forEach(([key, value]) => {
            const el = document.getElementById(`${key}-display`);
            if (el) el.innerText = value;
        });
    }

    // Portfolio logic
    async updatePortfolioView(force = false) {
        if (!this.dom.portfolioContainer) return;

        // Toggle Refresh button visibility
        if (this.dom.refreshBtn) {
            this.dom.refreshBtn.classList.toggle(
                'invisible-placeholder', 
                this.state.portfolioMode !== 'efficient_frontier'
            );
        }

        if (this.dom.portfolioTitle) {
            this.dom.portfolioTitle.innerText = this.state.portfolioMode.toUpperCase().replace(/_/g, ' ');
        }

        const url = `/get_portfolio_data?mode=${this.state.portfolioMode}&force_update=${force}`;
        
        try {
            const data = await this.apiRequest(url);
            const fig = JSON.parse(data.fig_data);
            const config = (this.state.portfolioMode === 'heatmap') ? { displayModeBar: false } : {};
            Plotly.react(this.dom.portfolioContainer, fig.data, fig.layout, config);
        } catch (err) {
            console.error("Portfolio Plot Error:", err);
        }
    }

    refreshSidebarUI() {
        if (this.state.currentMode === 'map-2dcorr' && this.state.secondaryTicker && this.dom.tickerTitle) {
            this.dom.tickerTitle.innerText = `${this.state.currentTicker} vs ${this.state.secondaryTicker}`;
        }

        this.dom.tickerRows.forEach(row => {
            const rowTicker = row.dataset.ticker;
            const indicator = row.querySelector('.indicator');

            row.classList.remove('active-ticker', 'secondary-ticker');
            if (indicator) indicator.innerText = '';

            if (rowTicker === this.state.currentTicker) {
                row.classList.add('active-ticker');
                if (indicator) indicator.innerText = '▶';
            } else if (this.state.currentMode === 'map-2dcorr' && rowTicker === this.state.secondaryTicker) {
                row.classList.add('secondary-ticker');
                if (indicator) indicator.innerText = 'II';
            }
        });
    }

    async handleExpandHistory() {
        if (!this.dom.startDateInput) return;
        const startDate = this.dom.startDateInput.value;
        if (!startDate) {
            alert("Please select a date first.");
            return;
        }

        if (!this.dom.expandBtn) return;
        const originalText = this.dom.expandBtn.innerHTML;
        
        this.dom.expandBtn.innerHTML = `<span class="loader"></span> Downloading...`;
        this.dom.expandBtn.disabled = true;

        // Use the class state for the current ticker
        const url = `/expand_history?ticker=${this.state.currentTicker}&start=${startDate}`;
        
        try {
            const data = await this.apiRequest(url);
            alert(data.message);
            // Refresh the current plot to show the new longer history
            this.updateView(); 
        } catch (err) {
            console.error("Expand Error:", err);
            alert("Failed to expand history.");
        } finally {
            this.dom.expandBtn.innerHTML = originalText;
            this.dom.expandBtn.disabled = false;
            //location.reload();
        }
    }

    registerEventListeners() {
        // Portfolio Tabs
        this.dom.portfolioTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.dom.portfolioTabs.forEach(t => t.classList.remove('portfolio-active-tab'));
                e.target.classList.add('portfolio-active-tab');
                this.state.portfolioMode = e.target.dataset.mode;
                this.updatePortfolioView();
            });
        });

        // Asset tabs
        this.dom.tickerTabs.forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.dom.tickerTabs.forEach(t => t.classList.remove('active-tab'));
                e.target.classList.add('active-tab');
                this.state.currentMode = e.target.dataset.mode;
                
                if (this.state.currentMode === 'map-2dcorr' && !this.state.secondaryTicker) {
                    const other = [...this.dom.tickerRows].find(r => r.dataset.ticker !== this.state.currentTicker);
                    this.state.secondaryTicker = other ? other.dataset.ticker : this.state.currentTicker;
                }
                this.refreshSidebarUI();
                this.updateView();
            });
        });

        // Ticker sidebar rows
        this.dom.tickerRows.forEach(row => {
            row.addEventListener('click', (e) => {
                const clicked = e.currentTarget.dataset.ticker;
                if (this.state.currentMode === 'map-2dcorr') {
                    if (clicked !== this.state.currentTicker) this.state.secondaryTicker = clicked;
                } else {
                    this.state.currentTicker = clicked;
                }
                this.refreshSidebarUI();
                this.updateView();
            });
        });

        // Refresh/Expansion Buttons
        if (this.dom.refreshBtn) {
            this.dom.refreshBtn.addEventListener('click', () => this.updatePortfolioView(true));
        }
        // History Expansion Button
        if (this.dom.expandBtn) {
            this.dom.expandBtn.addEventListener('click', () => this.handleExpandHistory());
        }
    }
}
