import { PortfolioController } from './portfolio/portfolio_controller.js';

document.addEventListener('DOMContentLoaded', () => {
    console.log("Initialising Research App...");
    window.activeApp = new PortfolioController({ selectedTicker: "{{ selected_ticker | default('', true) }}" });
});