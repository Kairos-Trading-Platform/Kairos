export class StrategyManager {
    constructor(app) {
        this.app = app;
        this.dom = {
            controls: document.getElementById('strategy-controls'),
            select: document.getElementById('strategy-select'),
            params: document.getElementById('strategy-params'),
            runBtn: document.getElementById('run-strategy-btn'),
            // Warning container for missing tickers
            warningBox: document.getElementById('strategy-warning-box'),
            warningList: document.getElementById('missing-tickers-list'),
            fixBtn: document.getElementById('fix-missing-btn')
        };
        this.currentSchema = [];
        this.portfolioTickers = []; // Will be populated from portfolio data
    }

    async init() {
        if (!this.dom.select) return;
        const strategies = await this.app.apiRequest('/strategies');
        this.dom.select.innerHTML = strategies
            .map(s => `<option value="${s.key}">${s.label}</option>`).join('');
        
        await this.loadPortfolioTickers(); // Load portfolio tickers for auto-fill
        
        this.dom.select.addEventListener('change', () => this.loadSchema());
        this.dom.runBtn.addEventListener('click', () => this.run());
        this.dom.fixBtn?.addEventListener('click', () => this.fixMissingTickers());

        await this.loadSchema();
    }

    // Fetch portfolio tickers to use as defaults
    async loadPortfolioTickers() {
        try {
            // Assumes you have an endpoint that returns current portfolio tickers
            // Adjust the endpoint as needed (e.g., /portfolio/tickers or extract from /api/background_check)
            const data = await fetch('/api/background_check').then(r => r.json());
            
            if (data.assets) {
                this.portfolioTickers = Object.keys(data.assets);
            } else if (data.portfolio?.stocks?.assets) {
                this.portfolioTickers = data.portfolio.stocks.assets.map(a => a.ticker);
            }
            
            console.log(`Loaded ${this.portfolioTickers.length} portfolio tickers:`, this.portfolioTickers);
        } catch (err) {
            console.warn('Could not load portfolio tickers for strategy defaults:', err);
            this.portfolioTickers = [];
        }
    }

    async loadSchema() {
        const key = this.dom.select.value;
        this.currentSchema = await this.app.apiRequest(`/strategies/${key}/schema`);
        
        // Auto-populate ticker fields with portfolio tickers
        this.dom.params.innerHTML = this.currentSchema.map(p => {
            let defaultValue = p.default;
            
            // If this is a ticker field and we have portfolio tickers, use them as default
            if ((p.type === 'ticker' || p.type === 'multi_ticker') && this.portfolioTickers.length > 0) {
                if (p.type === 'ticker') {
                    // Single ticker: use first portfolio ticker
                    defaultValue = this.portfolioTickers[0];
                } else if (p.type === 'multi_ticker') {
                    // Multi-ticker: use first 2-3 portfolio tickers
                    defaultValue = this.portfolioTickers.slice(0, 3).join(', ');
                }
            }
            
            return `
            <div class="form-group" data-param-key="${p.key}" data-param-type="${p.type}">
                <label>${p.label}</label>
                <input id="param_${p.key}" 
                       value="${Array.isArray(defaultValue) ? defaultValue.join(', ') : defaultValue}"
                       ${p.type === 'multi_ticker' ? 'placeholder="e.g., AMAT, LRCX"' : ''}>
                ${this.getPortfoliolHint(p)}
            </div>`;
        }).join('');
    }

    // Show hint about available portfolio tickers
    getPortfoliolHint(param) {
        if ((param.type === 'ticker' || param.type === 'multi_ticker') && this.portfolioTickers.length > 0) {
            return `
                <small class="hint-text" style="display: block; margin-top: 4px; color: #666; font-size: 0.85em;">
                    Available in portfolio: ${this.portfolioTickers.slice(0, 5).join(', ')}${this.portfolioTickers.length > 5 ? '...' : ''}
                </small>`;
        }
        return '';
    }

    collectParams() {
        const out = {};
        this.currentSchema.forEach(p => {
            const raw = document.getElementById(`param_${p.key}`).value;
            
            // Parse ticker arrays
            if (p.type === 'multi_ticker') {
                out[p.key] = raw.split(',').map(s => s.trim()).filter(Boolean);
            } else if (p.type === 'ticker') {
                out[p.key] = raw.trim();
            } else {
                // Other types: int, float, select, etc.
                out[p.key] = raw;
            }
        });
        // Strip dep_col out of any multi_ticker (indep_cols) field if it snuck in
        if (out.dep_col) {
            Object.entries(out).forEach(([key, val]) => {
                if (Array.isArray(val)) out[key] = val.filter(t => t !== out.dep_col);
            });
        }
        
        return out;
    }

    // Validate tickers before running
    validateTickers(params) {
        const missing = [];
        const requiredTickerFields = ['dep_col', 'indep_cols', 'tickers', 'symbols'];
        
        for (const field of requiredTickerFields) {
            const value = params[field];
            if (value) {
                const tickers = Array.isArray(value) ? value : [value];
                for (const ticker of tickers) {
                    if (!this.portfolioTickers.includes(ticker)) {
                        missing.push(ticker);
                    }
                }
            }
        }
        
        return missing;
    }

    // Show warning box for missing tickers
    showWarning(missingTickers, availableTickers) {
        if (!this.dom.warningBox) return;
        
        this.dom.warningBox.style.display = 'block';
        this.dom.warningList.innerHTML = missingTickers
            .map(t => `<li><code>${t}</code> not in live portfolio</li>`).join('');
        
        // Suggest closest available alternatives
        const suggestions = availableTickers.slice(0, 5);
        const suggestionHtml = suggestions.length > 0
            ? `<p><strong>Suggested alternatives:</strong> ${suggestions.join(', ')}</p>`
            : '';
        
        this.dom.warningBox.innerHTML = `
            <div style="background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; padding: 12px; margin-bottom: 10px;">
                <strong style="color: #856404;">⚠️ Missing ${missingTickers.length} ticker(s)</strong>
                <ul style="margin: 8px 0; padding-left: 20px; color: #856404;">
                    ${this.dom.warningList.innerHTML}
                </ul>
                ${suggestionHtml}
                <div style="margin-top: 10px;">
                    <button id="fix-missing-btn" style="
                        padding: 6px 12px;
                        background: #007BFF;
                        color: white;
                        border: none;
                        border-radius: 3px;
                        cursor: pointer;
                        font-size: 0.9em;">
                        Use Portfolio Tickers Instead
                    </button>
                    <button onclick="document.getElementById('strategy-warning-box').style.display='none'"
                            style="
                                padding: 6px 12px;
                                background: #6c757d;
                                color: white;
                                border: none;
                                border-radius: 3px;
                                cursor: pointer;
                                font-size: 0.9em;
                                margin-left: 8px;">
                        Ignore & Run Anyway
                    </button>
                </div>
            </div>
        `;
        
        // Re-bind click handler since we recreated the button
        document.getElementById('fix-missing-btn').addEventListener('click', () => this.fixMissingTickers(missingTickers));
    }

    hideWarning() {
        if (this.dom.warningBox) {
            this.dom.warningBox.style.display = 'none';
        }
    }

    // Auto-fix by replacing missing with available tickers
    fixMissingTickers(missingTickers) {
        const params = this.collectParams();
        const available = this.portfolioTickers.filter(t => !missingTickers.includes(t));
        
        if (available.length === 0) {
            alert('No portfolio tickers available to substitute!');
            return;
        }
        
        // Replace missing tickers in all fields
        for (const [key, value] of Object.entries(params)) {
            if (Array.isArray(value)) {
                params[key] = value.map(t => 
                    missingTickers.includes(t) ? available[0] : t
                );
            } else if (typeof value === 'string') {
                if (missingTickers.includes(value)) {
                    params[key] = available[0];
                }
            }
        }
        
        // Update the input fields with fixed values
        this.currentSchema.forEach(p => {
            if (params[p.key] !== undefined) {
                const input = document.getElementById(`param_${p.key}`);
                if (input) {
                    input.value = Array.isArray(params[p.key]) 
                        ? params[p.key].join(', ') 
                        : params[p.key];
                }
            }
        });
        
        this.hideWarning();
        alert(`Replaced missing tickers with ${available[0]} (and others)`);
    }

    async run() {
        const key = this.dom.select.value;
        const params = this.collectParams();
        
        // VALIDATION: Check for missing tickers
        const missing = this.validateTickers(params);
        
        if (missing.length > 0) {
            // Show warning instead of blocking
            this.showWarning(missing, this.portfolioTickers);
            
            // Pause execution - user must choose: fix or ignore
            return; // Don't proceed until they take action
        }
        
        this.hideWarning();
        
        try {
            const data = await this.app.apiRequest(`/strategies/${key}/run`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(params)
            });
            
            // Check for server-side warnings/errors in response
            if (data.error || data.warning) {
                if (this.app.dom?.plotMessage) {
                    this.app.dom.plotMessage.style.color = data.error ? 'red' : 'orange';
                    this.app.dom.plotMessage.innerText = data.error || data.warning;
                    this.app.dom.plotMessage.style.display = 'block';
                }
                Plotly.purge(this.app.dom.portfolioContainer);
                return;
            }
            
            this.app.renderPlot('portfolio-plot-container', data.fig_data);
        } catch (err) {
            console.error("Strategy execution error:", err);
            alert("Strategy failed. Check console for details.");
        }
    }

    toggle(show) {
        this.dom.controls.style.display = show ? 'block' : 'none';
    }
}