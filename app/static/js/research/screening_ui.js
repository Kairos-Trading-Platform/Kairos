export class ScreeningUI {
    constructor(app) {
        this.app = app;
        this.dom = {
            depInput: document.getElementById('screening-dep'),
            indepInput: document.getElementById('screening-indep'),
            testBtn: document.getElementById('screening-test-btn'),
            resultBox: document.getElementById('screening-result'),
            plotContainer: document.getElementById('screening-plot-container'),
            table: document.getElementById('screening-history-table'),
        };
        this.init();
    }

    init() {
        this.dom.testBtn?.addEventListener('click', () => this.runTest());
        this.refreshHistory();
    }

    async runTest() {
        const dep_col = this.dom.depInput.value.trim();
        const indep_cols = this.dom.indepInput.value.split(',').map(s => s.trim()).filter(Boolean);
        if (!dep_col || !indep_cols.length) { alert('Enter dep_col and at least one indep_col.'); return; }

        const data = await this.app.apiRequest('/research/screening/test', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dep_col, indep_cols })
        });

        this.dom.resultBox.style.color = data.status === 'passed' ? 'green' : 'red';
        this.dom.resultBox.innerText = data.status === 'passed'
            ? `Passed: ${JSON.stringify(data.summary)}`
            : `Failed at [${data.stage}]: ${data.reason}`;

        if (data.fig_data) {
            this.app.renderPlot('screening-plot-container', data.fig_data);
        } else {
            Plotly.purge(this.dom.plotContainer);
        }
        this.refreshHistory();
    }

    async refreshHistory() {
        const records = await this.app.apiRequest('/research/screening');
        this.dom.table.querySelector('tbody').innerHTML = records.slice().reverse().map(r => `
            <tr class="screening-row" data-plot='${(r.plot || '').replace(/'/g, "&apos;")}' style="cursor:pointer;">
                <td>${r.dep_col} / ${r.indep_col}</td>
                <td style="color:${r.status === 'passed' ? 'green' : 'red'}">${r.status}</td>
                <td>${r.stage || ''}</td>
                <td>${r.reason || ''}</td>
            </tr>`).join('');

        this.dom.table.querySelectorAll('.screening-row').forEach(row => {
            row.addEventListener('click', () => {
                const plot = row.dataset.plot;
                if (plot) this.app.renderPlot('screening-plot-container', plot);
            });
        });
    }
}