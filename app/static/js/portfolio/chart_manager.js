export class ChartManager {
    /**
     * @param {PortfolioController} app - Reference to the main controller
     */
    constructor(app) {
        this.app = app;
        // Configuration for the marker colors to keep them consistent
        this.colors = ['#3498db', '#e74c3c', '#f1c40f', '#2ecc71', '#9b59b6'];
    }

    /**
     * Main entry point called by the controller
     */
    update(data, subPortfolio, assetType) {
        console.log(`ChartManager: Updating charts for ${assetType}`);
        
        // Update Asset Allocation and Sector Diversification
        this.refreshPieCharts(data, subPortfolio, assetType);

        // Update the Monthly Income Bar Chart (specific to the portfolio view)
        const plotDiv = document.getElementById('monthly-income-plot');
        if (plotDiv && data.income_plot) {
            this.app.renderPlot('monthly-income-plot', data.income_plot);
        }
    }

    refreshPieCharts(data, subPortfolio, assetType) {
        console.log(`refreshPieCharts called for asset type ${assetType}`);
        const manager = data.portfolio;
        
        // Prepare data for Global Allocation
        const allTotals = {
            stocks: manager.stocks.total_market_value || 0,
            crypto: manager.crypto.total_market_value || 0,
            interest: manager.interest.total_market_value || 0,
            cash: manager.summary.free_cash || 0
        };

        const categories = ['stocks','crypto','interest','cash']; //TODO add assets
        let allocLabels = [];
        let allocValues = [];

        categories.forEach(cat => {
            const val = allTotals[cat];
            if (val > 0) {
                allocLabels.push(cat.charAt(0).toUpperCase() + cat.slice(1));
                allocValues.push(val);
                console.log("category: ", cat);
                console.log("val: ", val);
            }
        });

        // Update Global Allocation Chart
        this.renderPieChart(
            'asset-allocation-chart', 
            allocLabels, 
            allocValues, 
            'Global allocation'
        );

        // Update Sector Diversification (Only if on Stocks tab and data exists)
        if (assetType === 'stocks' && subPortfolio.sectors.labels && subPortfolio.sectors) {
            this.renderPieChart(
                'sector-chart', 
                subPortfolio.sectors.labels, 
                subPortfolio.sectors.values, 
                'Sector diversification'
            );
        }
    }

    /**
     * Internal helper to build the Plotly Pie trace and layout
     */
    renderPieChart(elementId, labels, values, chartTitle) {
        if (!document.getElementById(elementId)) return;

        const total = values.reduce((sum, v) => sum + v, 0); // Calculate total for percentage calculation
        const customText = labels.map((_, i) => {
            const percent = ((values[i] / total) * 100).toFixed(2);
            return `${percent}%`;
        });

        const trace = [{
            values: values,
            labels: labels,
            type: 'pie',
            hole: 0.4,
            text: customText,
            textinfo: "label+text",
            textposition: "outside",
            automargin: false,
            domain: {
                x: [0.15, 0.85],
                y: [0.15, 0.85]  // Gives space at bottom and top (for title)
            },
            hovertemplate: "<b>%{label}</b><br>%{text}<br><extra></extra>",
            hoverlabel: {
                align: 'center'
            },
            marker: { colors: this.colors }
        }];

        const layout = {
            title: {
                text: chartTitle,
                font: { color: '#ffffff', size: 18 },
                y: 0.95
            },
            autosize: true,
            margin: { t: 40, b: 20, l: 20, r: 20 },
            showlegend: false,
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#ffffff' }
        };

        const config = {
            responsive: true,
            displayModeBar: false
        };

        Plotly.react(elementId, trace, layout, config);
    }
}
