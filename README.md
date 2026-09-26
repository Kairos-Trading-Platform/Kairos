# Kairos
_Last updated: 2026-09-26_


## Finance platform for portfolio monitoring and quantitative research.

There are three main features at the moment:
  1) A balance sheet for personal finance
  2) A portofolio monitoring
  3) A research dashboard

The balance sheet contains all incomes and expenses over the year.

The portfolio monitoring gathers all long positions that you have. It shows various equity metrics (P/E, PEG, dividends, etc...) and how diversified you are accross assets. There is a colour code to help growth and value investors alike (green is good, red is bad). Note that the data is pulled from Yahoo Finance, so add tickers that they use.

The research dashboard contains several features used for understanding your assets: historical prices with SMAs, log return distributions, correlations.

Use ``flask run`` to run it. Make sure all the dependencies in `requirements.txt` are installed in your Python environment.
