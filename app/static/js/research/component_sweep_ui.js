export class ComponentSweepUI {
    constructor() {
        this.select = document.getElementById('component-select');
        this.paramsDiv = document.getElementById('component-params');
        this.init();
    }
    async init() {
        const components = await fetch('/research/components').then(r => r.json());
        components.forEach(c => this.select.add(new Option(c.label, c.key)));
        this.select.addEventListener('change', () => this.onSelect());
        document.getElementById('run-sweep-btn').addEventListener('click', () => this.runSweep());
    }
    async onSelect() {
        const key = this.select.value;
        if (!key) { this.paramsDiv.style.display = 'none'; return; }
        const schema = await fetch(`/research/components?component=${key}`).then(r => r.json());
        this.paramsDiv.innerHTML = schema.map(p => `
            <label>${p.key} (default: ${p.default})
                <input type="text" data-field="${p.key}" placeholder="comma-separated test values">
            </label>`).join('');
        this.paramsDiv.style.display = 'block';
    }
    async runSweep() {
        const mode = document.querySelector('input[name="sweep-mode"]:checked').value;
        const param_values = {};
        this.paramsDiv.querySelectorAll('input[data-field]').forEach(inp => {
            if (inp.value.trim()) {
                param_values[inp.dataset.field] = inp.value.split(',').map(v => v.trim());
            }
        });
        const body = { dep_col: window.currentDep, indep_cols: window.currentIndep, param_values, mode };
        const data = await fetch('/research/components/sweep', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body)
        }).then(r => r.json());
        // render data.results as a table
    }
}