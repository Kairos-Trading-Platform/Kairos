export class PortfolioUIManager {
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