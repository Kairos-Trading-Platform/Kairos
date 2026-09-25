import { StrategyManager } from '/static/js/research/strategy_manager.js'
import { FinAppBase } from '/static/js/core/base.js';

export class ResearchController extends FinAppBase {
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
        this.strategyManager = new StrategyManager(this);
        this.strategyManager.init();

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

                this.strategyManager.toggle(this.state.portfolioMode === 'strategies');
                if (this.state.portfolioMode !== 'strategies') this.updatePortfolioView();
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