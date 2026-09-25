import { NotificationManager } from '/static/js/notifications/notification_manager.js';

export class FinAppBase {
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
            if (!response.ok) {
            const errorText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errorText}`);
            }
            return await response.json();
        } catch (err) {
            console.error(`API Error (${url}):`, err);
            // Show user-friendly error
            if (this.app?.dom?.plotMessage) {
                this.app.dom.plotMessage.style.color = 'red';
                this.app.dom.plotMessage.innerText = `API Error: ${err.message}`;
                this.app.dom.plotMessage.style.display = 'block';
            } else if (!url.includes('/background_check')) {
                // Don't alert on background sync failures
                alert(`Request failed: ${err.message}`);
            }
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
