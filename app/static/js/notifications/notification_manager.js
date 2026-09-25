export class NotificationManager {
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