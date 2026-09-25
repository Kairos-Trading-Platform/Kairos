export class PortfolioController extends FinAppBase {
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
        document.addEventListener('DOMContentLoaded', () => {
            new ResearchController({ selectedTicker: "{{ selected_ticker | default('', true) }}" });
            new ComponentSweepUI();
        });
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
