export class TickerManager {
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
