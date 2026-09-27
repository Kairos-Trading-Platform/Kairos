export class FamilyTabController {
    /**
     * @param {StrategyManager} strategyManager - existing instance, untouched
     * @param {ScreeningUI} screeningUI - existing instance, untouched
     */
    constructor(strategyManager, screeningUI) {
        this.strategyManager = strategyManager;
        this.screeningUI = screeningUI;
        this.tabsContainer = document.getElementById('strategy-family-tabs');
        this.screeningBlock = document.getElementById('screening-controls');
        this.allStrategies = [];
        this.activeFamily = null;
    }

    async init() {
        this.allStrategies = await this.strategyManager.app.apiRequest('/strategies');
        this.strategyManager.setStrategies(this.allStrategies);   // built once, all families
        this._renderTabs();
        const firstFamily = this.families()[0];
        if (firstFamily) this.selectFamily(firstFamily);
    }

    families() {
        return [...new Set(this.allStrategies.map(s => s.family || 'Other'))];
    }

    _renderTabs() {
        this.tabsContainer.innerHTML = this.families().map(f =>
            `<div class="tab" data-family="${f}">${f}</div>`
        ).join('');
        this.tabsContainer.querySelectorAll('.tab').forEach(tab =>
            tab.addEventListener('click', () => this.selectFamily(tab.dataset.family))
        );
    }

    selectFamily(family) {
        this.activeFamily = family;

        this.tabsContainer.querySelectorAll('.tab').forEach(tab =>
            tab.classList.toggle('active-tab', tab.dataset.family === family)
        );

        const filtered = this.allStrategies.filter(s => (s.family || 'Other') === family);
        this.strategyManager.updateQuickPicks(filtered);

        // Jump the select to the first strategy in this family
        const select = this.strategyManager.dom.select;
        const firstOption = select.querySelector(`optgroup[label="${family}"] option`);
        if (firstOption) {
            select.value = firstOption.value;
            this.strategyManager.loadSchema();
        }

        this.screeningBlock.style.display = (family === 'Kalman') ? 'block' : 'none';
    }
}