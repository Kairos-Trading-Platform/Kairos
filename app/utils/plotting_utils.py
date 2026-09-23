import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from itertools import combinations
from scipy.spatial.distance import cdist
from scipy import stats
import logging
logger = logging.getLogger(__name__)


### Private helpers

def _filter_by_date(df: pd.DataFrame, 
                    start_date=None, end_date=None) -> pd.DataFrame:
    if start_date:
        ts = pd.to_datetime(start_date) # Ensure timezone compatibility
        if df.index.tz:
            ts = ts.tz_localize(df.index.tz)
        df = df[df.index >= ts]
    if end_date:
        ts = pd.to_datetime(end_date)
        if df.index.tz:
            ts = ts.tz_localize(df.index.tz)
        df = df[df.index <= ts]
    return df


def _add_rolling_traces(fig: go.Figure, series: pd.Series, 
                        name: str, windows: list[int]) -> None:
    """Adds SMA traces to an existing figure in-place."""
    for window in windows:
        rolling = series.rolling(window=window).mean()
        fig.add_trace(go.Scatter(
            x=rolling.index.tolist(),
            y=rolling.values.tolist(),
            mode='lines',
            name=f"{name} ({window}d SMA)",
            line=dict(dash='dash', width=1.5),
            hovertemplate=f"{name} {window}d SMA: %{{y:.2f}}<extra></extra>"
        ))

def _base_layout(**overrides) -> dict:
    """Shared layout defaults for all charts."""
    base = dict(
        template='plotly_white',
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation='h', yanchor='bottom', 
                    y=1.02, xanchor='right', x=1),
        hovermode='x unified',
    )
    base.update(overrides)
    return base

def _price_layout(title: str) -> dict:
    return _base_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Price',
        dragmode='pan',
        xaxis=dict(rangeslider=dict(visible=True), type='date', fixedrange=True),
        yaxis=dict(type='log', autorange=True, 
                   title='Price - Log Scale', fixedrange=False)
    )

def _trends_layout(title: str) -> dict:
    return _base_layout(
        title=title,
        xaxis_title='Date',
        xaxis=dict(rangeslider=dict(visible=True), type='date'),
        yaxis=dict(autorange=True, title='Trends Index')
    )

def _merge_overlapping_labels(df_lsa: pd.DataFrame,
                               threshold: float = 0.02) -> dict:
    """
    Returns a dict mapping each term to a merged label string
    if it overlaps with nearby terms in n-dimensional component space.
    Works for any number of components.
    """

    coords = df_lsa.values  # shape (n_terms, n_components)

    # Normalize each axis by its range to make threshold scale-independent
    ranges = np.ptp(coords, axis=0)
    ranges[ranges == 0] = 1  # avoid division by zero for constant columns
    normalized = coords / ranges

    distances = cdist(normalized, normalized)
    np.fill_diagonal(distances, np.inf)

    terms = df_lsa.index.tolist()
    merged = {}
    assigned = set()

    for i, term in enumerate(terms):
        if i in assigned:
            continue
        close = [i] + [j for j in range(len(terms))
                       if distances[i, j] < threshold and j not in assigned]
        group_label = ', '.join(terms[k] for k in close)
        for k in close:
            merged[terms[k]] = group_label
            assigned.add(k)

    return merged

### LSA charts

def create_lsa_scatter(df_lsa: pd.DataFrame, seed: int = 42) -> go.Figure:
    cols = df_lsa.columns.tolist()

    if len(cols) == 3:
        return _lsa_scatter_3d(df_lsa)
    else:
        return _lsa_scatter_2d_dropdown(df_lsa, seed)

def _lsa_scatter_3d(df_lsa: pd.DataFrame) -> go.Figure:
    col_x, col_y, col_z = df_lsa.columns[:3]
    merged_labels = _merge_overlapping_labels(df_lsa)
    hover_texts = [
        f"<b>{merged_labels[term]}</b><br>"
        f"({df_lsa.loc[term, col_x]:.4f}, "
        f"{df_lsa.loc[term, col_y]:.4f}, "
        f"{df_lsa.loc[term, col_z]:.4f})"
        for term in df_lsa.index
    ]
    # Avoid duplicate text on the plot
    seen_groups = set()
    plot_labels = []
    for term in df_lsa.index:
        group = merged_labels[term]
        if group in seen_groups:
            plot_labels.append('')  # empty / group already labelled
        else:
            plot_labels.append(group)
            seen_groups.add(group)

    fig = go.Figure(go.Scatter3d(
        x=df_lsa[col_x].tolist(),
        y=df_lsa[col_y].tolist(),
        z=df_lsa[col_z].tolist(),
        mode='markers+text',
        text=plot_labels,
        textfont=dict(size=11),
        marker=dict(size=5, color='#3498db', opacity=0.8),
        hovertemplate='%{customdata}<extra></extra>',
        customdata=hover_texts
    ))
    fig.update_layout(
        title='LSA Term Space (3D)',
        scene=dict(
            xaxis_title=col_x,
            yaxis_title=col_y,
            zaxis_title=col_z
        ),
        font=dict(color='#043657'),
    )
    return fig

def _lsa_scatter_2d_dropdown(df_lsa: pd.DataFrame, seed: int = 42) -> go.Figure:
    cols = df_lsa.columns.tolist()
    rng = np.random.default_rng(seed)
    fig = go.Figure()
    merged_labels = _merge_overlapping_labels(df_lsa)

    seen_groups = set()
    plot_labels = []
    for term in df_lsa.index:
        group = merged_labels[term]
        if group in seen_groups:
            plot_labels.append('')  # empty / group already labelled
        else:
            plot_labels.append(group)
            seen_groups.add(group)

    # Generate all pairwise combinations
    pairs = list(combinations(cols, 2))

    for i, (col_x, col_y) in enumerate(pairs):
        hover_texts = [
            f"<b>{merged_labels[term]}</b><br>"
            f"({df_lsa.loc[term, col_x]:.5f}, "
            f"{df_lsa.loc[term, col_y]:.5f})"
            for term in df_lsa.index
        ]
        jitter_scale = (df_lsa[col_x].max() - df_lsa[col_x].min()) * 0.02
        label_x = df_lsa[col_x] + rng.uniform(-jitter_scale, jitter_scale, len(df_lsa))
        label_y = df_lsa[col_y] + rng.uniform(-jitter_scale, jitter_scale, len(df_lsa))

        fig.add_trace(go.Scatter(
            x=df_lsa[col_x].tolist(),
            y=df_lsa[col_y].tolist(),
            mode='markers',
            marker=dict(size=10, color='#3498db', opacity=0.8),
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
            visible=(i == 0),  # only first pair visible initially
            showlegend=False,
            name=f'{col_x} vs {col_y}'
        ))
        fig.add_trace(go.Scatter(
            x=label_x.tolist(),
            y=label_y.tolist(),
            mode='text',
            text=plot_labels,
            textfont=dict(size=11, color='#043657'),
            textposition='top center',
            hoverinfo='skip',
            visible=(i == 0),
            showlegend=False,
            name=f'labels_{col_x}_{col_y}'
        ))

    # Build dropdown — each pair shows its two traces (marker + label)
    buttons = []
    for i, (col_x, col_y) in enumerate(pairs):
        visibility = [False] * (len(pairs) * 2)
        visibility[i * 2] = True      # marker trace
        visibility[i * 2 + 1] = True  # label trace
        buttons.append(dict(
            label=f'{col_x} vs {col_y}',
            method='update',
            args=[
                {'visible': visibility},
                {'xaxis': {'title': col_x},
                 'yaxis': {'title': col_y},
                 'title': f'LSA: {col_x} vs {col_y}'}
            ]
        ))

    fig.update_layout(
        updatemenus=[dict(
            buttons=buttons,
            direction='down',
            x=0.0,
            y=1.1,
            showactive=True
        )],
        title=f'LSA: {pairs[0][0]} vs {pairs[0][1]}',
        xaxis_title=pairs[0][0],
        yaxis_title=pairs[0][1],
        font=dict(color='#043657'),
    )
    return fig

### Time series charts

def create_trends_chart(interest_over_time_df,terms=None, start_date=None, 
                        end_date=None, rolling_windows=None) -> go.Figure:
    """
    Line chart showing Google Trends interest over time for multiple terms.

    Args:
        interest_over_time_df (pd.DataFrame): DataFrame with 'Date' index and terms as columns.
        terms (list): List of terms to plot. If None, plot all columns.
        start_date (str/datetime): Start date for filtering (e.g., '2024-01-01').
        end_date (str/datetime): End date for filtering.
        rolling_windows (list): List of integers for rolling means, e.g., [7, 30].

    Returns:
        go.Figure: The Plotly figure object.
    """

    df = _filter_by_date(interest_over_time_df.copy(), start_date, end_date)

    # Filter terms if specified
    if terms:
        df = df[terms]

    # Create the figure
    fig = go.Figure()

    # Add a line for each term
    for term in df.columns:
        series = df[term].dropna()
        if series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series.index.tolist(),
            y=series.values.tolist(),
            mode='lines',
            name=term,
            hovertemplate=f"<b>{term}</b><br>Trends Index: %{{y:.2f}}<extra></extra>"
        ))
        if rolling_windows:
            _add_rolling_traces(fig, series, term, rolling_windows)

    # Update layout
    fig.update_layout(**_trends_layout('Interest over time'))

    return fig

def create_price_chart(stocks_data, start_date=None, end_date=None, rolling_windows=None) -> go.Figure:
    """
    Line chart showing price history for multiple assets.
    
    Args:
        stocks_data (pd.DataFrame): DataFrame with 'Date' index and tickers as columns.
        start_date (str/datetime): Start date for filtering (e.g., '2024-01-01').
        end_date (str/datetime): End date for filtering.
        rolling_windows (list): List of integers for rolling means, e.g., [50, 200].
    Returns:
        go.Figure: The Plotly figure object.
    """

    df = _filter_by_date(stocks_data.copy(), start_date, end_date)    

    # Create the figure
    fig = go.Figure()

    # Add a line for each ticker
    for ticker in df.columns:
        series = df[ticker].dropna()
        if series.empty:
            continue
        fig.add_trace(go.Scatter(
            x=series.index.tolist(),
            y=series.values.tolist(),
            mode='lines',
            name=ticker,
            hovertemplate=f"<b>{ticker}</b><br>Price: %{{y:.2f}}<extra></extra>"
        ))
        if rolling_windows:
            _add_rolling_traces(fig, series, ticker, rolling_windows)

    # Update layout
    fig.update_layout(**_price_layout('Price history'))
    
    return fig

### Portfolio charts

def plot_efficient_frontier_and_portfolios(
    results,
    asset_analysers
):
    """
    Plots the efficient frontier, Monte Carlo simulations, individual stocks,
    and optimised portfolios (MVP, Sharpe, Sortino, MVSK).

    Args:
        static_results (dict): Results from static optimisation.
        dynamic_results (dict): Results from dynamic optimisation.
        individual_stock_metrics (list): List of dictionaries with individual stock metrics.
        portfolio_tickers (list): List of ticker symbols for the assets.
        static_portfolio_points_raw_mc (list): List of dictionaries for static Monte Carlo portfolios.
        dynamic_portfolio_points_raw_mc (list): List of dictionaries for dynamic Monte Carlo portfolios.
        output_dir (str): The directory to save the plot.
        feature_toggles (dict): Dictionary of feature toggles.
        num_assets (int): Number of assets in the portfolio.
    """
    fig = go.Figure()
    tickers = list(asset_analysers.keys())
    
#    RUN_STATIC_PORTFOLIO = feature_toggles['RUN_STATIC_PORTFOLIO']
    RUN_STATIC_PORTFOLIO = True
#    RUN_DYNAMIC_PORTFOLIO = feature_toggles['RUN_DYNAMIC_PORTFOLIO']
#    RUN_EQUAL_WEIGHTED_PORTFOLIO = feature_toggles['RUN_EQUAL_WEIGHTED_PORTFOLIO']
#    RUN_MONTE_CARLO_SIMULATION = feature_toggles['RUN_MONTE_CARLO_SIMULATION']
#    RUN_MVO_OPTIMISATION = feature_toggles['RUN_MVO_OPTIMISATION']
    RUN_MVO_OPTIMISATION = True
#    RUN_SHARPE_OPTIMISATION = feature_toggles['RUN_SHARPE_OPTIMISATION']
#    RUN_SORTINO_OPTIMISATION = feature_toggles['RUN_SORTINO_OPTIMISATION']
#    RUN_MVSK_OPTIMISATION = feature_toggles['RUN_MVSK_OPTIMISATION']

#    plt.figure(figsize=(14, 8)) # Larger figure for more elements

#    # Plot all Monte-Carlo-simulated portfolio combinations (lighter color, background)
#    if RUN_MONTE_CARLO_SIMULATION:
#        if RUN_STATIC_PORTFOLIO and static_portfolio_points_raw_mc:
#            plt.scatter([p['std_dev'] * 100 for p in static_portfolio_points_raw_mc],
#                        [p['return'] * 100 for p in static_portfolio_points_raw_mc],
#                        color='blue', marker='o', s=10, alpha=0.5, # More transparent
#                        label='Monte Carlo portfolio combinations (Static)')
#        if RUN_DYNAMIC_PORTFOLIO and dynamic_portfolio_points_raw_mc and dynamic_results['dynamic_covariance_available']:
#            plt.scatter([p['std_dev'] * 100 for p in dynamic_portfolio_points_raw_mc],
#                        [p['return'] * 100 for p in dynamic_portfolio_points_raw_mc],
#                        color='red', marker='o', s=10, alpha=0.5, # More transparent
#                        label='Monte Carlo portfolio combinations (Dynamic)')
    
    # Plot the Efficient Frontier line (Static Covariance)
#    if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets > 20 and static_results['mvp'] and static_results['efficient_frontier_std_devs']:
#        plt.plot([s * 100 for s in static_results['efficient_frontier_std_devs']],
#                 [r * 100 for r in static_results['efficient_frontier_returns']],
#                 color='blue', linestyle='-', linewidth=2, label='Efficient frontier (Static)')

    # Extract sub-dictionaries for readability
    opt_res = results.get("optimisation")
    cml_res = results.get("cml")

    if RUN_STATIC_PORTFOLIO and RUN_MVO_OPTIMISATION and opt_res and opt_res['efficient_frontier_std_devs']:
        # Create a list of hover strings for each point
        hover_texts = []
        for w_set in opt_res.get('efficient_frontier_weights', []):
            # Only show assets with a weight > 0.5% to keep it clean
            weight_str = "<br>".join([
                f"{tickers[i]}: {w*100:.1f}%" 
                for i, w in enumerate(w_set) if w > 0.005
            ])
            hover_texts.append(weight_str)
        
        fig.add_trace(go.Scatter(
            x=[s * 100 for s in opt_res['efficient_frontier_std_devs']],
            y=[r * 100 for r in opt_res['efficient_frontier_returns']],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Efficient frontier (Static)',
            customdata=hover_texts,
            hovertemplate=(
                "<b>Efficient Portfolio</b><br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<br>" +
                "------------------<br>" +
                "%{customdata}%<extra></extra>"
            )
        ))

#    # Plot the Efficient Frontier line (Dynamic Covariance)
#    if RUN_DYNAMIC_PORTFOLIO and RUN_MVO_OPTIMISATION and num_assets > 20 and dynamic_results['mvp'] and dynamic_results['efficient_frontier_std_devs'] and dynamic_results['dynamic_covariance_available']:
#        plt.plot([s * 100 for s in dynamic_results['efficient_frontier_std_devs']],
#                 [r * 100 for r in dynamic_results['efficient_frontier_returns']],
#                 color='red', linestyle='-', linewidth=2, label='Efficient frontier (Dynamic)')

    # Plot CML
        fig.add_trace(go.Scatter(
            x=[v * 100 for v in cml_res['vols']],
            y=[r * 100 for r in cml_res['returns']],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='CML'
        ))

        # Mark the tangency portfolio point
        t_vol = cml_res['tangency_vol']
        t_ret = cml_res['tangency_ret']
        t_weights = cml_res['tangency_weights']

        t_hover = "<br>".join([
            f"{tickers[i]}: {w*100:.1f}%" 
            for i, w in enumerate(t_weights) if w > 0.005
        ])

        fig.add_trace(go.Scatter(
            x=[t_vol * 100],
            y=[t_ret * 100],
            mode='markers',
            marker=dict(color='red', size=10, symbol='star'),
            name='Tangency portfolio',
            customdata=[t_hover],
            hovertemplate=(
                "<b>Tangency portfolio</b><br>" +
                "Volatility: %{x:.2f}%<br>" +
                "Return: %{y:.2f}%<extra></extra>"+
                "------------------<br>" +
                "%{customdata}<extra></extra>"
            )
        ))

    # Plot individual assets in the return/std space
    colors = px.colors.qualitative.Dark24  # or any palette you prefer
    all_vols = []
    all_rets = []

    for i, (ticker, analyser) in enumerate(asset_analysers.items()):
        ret = analyser.annual_return
        vol = analyser.annualised_volatility
        all_vols.append(vol)
        all_rets.append(ret)

        fig.add_trace(go.Scatter(
            x=[vol * 100],
            y=[ret * 100],
            mode='markers+text',
            marker=dict(size=12, color=colors[i % len(colors)], line=dict(width=1.5, color='black')),
            text=[ticker],
            textposition='top center',
            name='',
            showlegend=False
        ))


    if RUN_STATIC_PORTFOLIO:
#        # Plot the EWP (Static)
#        if RUN_EQUAL_WEIGHTED_PORTFOLIO and static_results['ewp'] and static_results['ewp']['success']:
#            plt.scatter(static_results['ewp']['Volatility'] * 100, static_results['ewp']['Return'] * 100,
#                        marker='p', s=200, color='darkblue', edgecolor='darkblue', alpha=0.3, linewidth=1.5,
#                        label=f"EWP (Static), Sharpe ratio={static_results['ewp']['Sharpe Ratio']:.2}")
                        
        # Plot the MVP (Static)
        if RUN_MVO_OPTIMISATION and opt_res['mvp'] and opt_res['mvp']['success']:
            mvp_metrics = opt_res['mvp']['metrics']
            fig.add_trace(go.Scatter(
                x=[mvp_metrics['Volatility'] * 100],
                y=[mvp_metrics['Return'] * 100],
                mode='markers',
                marker=dict(size=16, symbol='star', color='darkblue', opacity=0.7, line=dict(width=1.5, color='darkblue')),
                name='MV portfolio'
            ))
            
    max_v = max(all_vols) * 110 if all_vols else 30
    min_r = min(all_rets) * 90 if all_rets else 0
    max_r = max(all_rets) * 110 if all_rets else 20

    fig.update_layout(
        xaxis=dict(title="Annualised Volatility (%)", range=[0, max_v]),
        yaxis=dict(title="Annualised Return (%)", range=[min_r, max_r]),
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def create_income_plot(income_data, title="Expected monthly income"):
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    source_colors = {
        'stocks':   '#28a745',  # green - dividends
        'crypto':   '#f39c12',  # orange - staking rewards
        'interest': '#3498db',  # blue - interest rewards
    }
    source_labels = {
        'stocks': 'Dividends',
        'crypto': 'Staking',
        'interest': 'Interests',
    }

    # Create Plotly figure
    # fig = go.Figure(data=[
    #     go.Bar(
    #         x=months,
    #         y=income_data['payouts'],
    #         marker_color=['#28a745' if amount > 0 else '#cccccc' for amount in income_data['payouts']],
    #         hovertext=income_data['details'],
    #         hovertemplate='%{hovertext}<extra></extra>', # <extra></extra> removes the secondary 'trace' box
    #         name=title
    #     )
    # ])

    # # Update layout for a non-static look
    # fig.update_layout(
    #     title=title + ' (€)',
    #     xaxis_title='Month',
    #     hovermode="x unified",
    #     margin=dict(l=20, r=20, t=50, b=20)
    # )

    fig = go.Figure()

    for source, color in source_colors.items():
        payouts = income_data.get(source)
        if not payouts or not any(payouts):
            continue
        fig.add_trace(go.Bar(
            x=months,
            y=payouts,
            name=source_labels[source],
            marker_color=color,
            hovertemplate=f'{source_labels[source]}: €%{{y:.2f}}<extra></extra>',
        ))

    fig.update_layout(**_base_layout(
        title=title + ' (€)',
        barmode='stack',
        xaxis_title='Month',
        hovermode='x unified',
    ))
    
    return fig

### Analysis charts

def plot_correlation_heatmap(correlation_matrix):
    """
    Plot a heatmap of the correlation matrix.

    Args:
        correlation_matrix (pd.DataFrame): The correlation matrix of stock returns.
    """
    logger.debug("plot_correlation_heatmap called")

    matrix_values = correlation_matrix.round(3).values.tolist()
    tickers = correlation_matrix.columns.tolist()
    
    heatmap = go.Heatmap(
        z=matrix_values,
        x=tickers,
        y=tickers,
        colorscale='RdBu_r',
        zmin=-1,
        zmax=1,
        text=[[f"{v:.2f}" for v in row] for row in matrix_values],
        hovertemplate='x: %{x}<br>y: %{y}<br>Correlation: %{z}<extra></extra>',
        colorbar=dict(title='Correlation')
    )
    
    fig = go.Figure(data=[heatmap])
    
    fig.update_layout(
        autosize=True,
        template="plotly_white",
        margin=dict(l=40, r=40, t=10, b=40),
        coloraxis_colorbar=dict(
            thickness=20,
            len=0.8,
            y=0.5,
            yanchor='middle',
            x=1,  # right at edge of plot domain
            ticks='outside',
            outlinewidth=1,
            outlinecolor='black',
        )
    )

    fig.update_xaxes(
        type='category',
        tickson='boundaries',
        constrain='domain',
        ticks='outside',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        tickangle=45,
        automargin=True
    )

    fig.update_yaxes(
        type='category',
        tickson='boundaries',
        constrain='domain',
        autorange='reversed',
        ticks='outside',
        showline=True,
        linewidth=1,
        linecolor='black',
        mirror=True,
        automargin=True
    )

    logger.debug("plot_correlation_heatmap out ")
    return fig
    
def create_2d_correlation_map(stocks_data_ticker1, stocks_data_ticker2):
    """
    Correlation stock vs stock.
    """
    # TODO extend that to other assets
    stocks1 = stocks_data_ticker1.copy()
    stocks2 = stocks_data_ticker2.copy()
    
    # Get name
    name1 = stocks1.name if hasattr(stocks1, 'name') else stocks1.columns[0]
    name2 = stocks2.name if hasattr(stocks2, 'name') else stocks2.columns[0]
    
    # Compute log returns
    returns1 = np.log(stocks1 / stocks1.shift(1))
    returns2 = np.log(stocks2 / stocks2.shift(1))
    
    # Join them into a single DataFrame
    map2d = pd.concat([returns1, returns2], axis=1, join='inner').dropna()
    
    if map2d.empty:
        return go.Figure().add_annotation(text="No overlapping data", showarrow=False)
        
    # Extract values
    x_data = map2d.iloc[:, 0].values
    y_data = map2d.iloc[:, 1].values
    
    fig = go.Figure()

    # Add Scatter Points
    fig.add_trace(go.Scatter(
        x=x_data.tolist(), # Convert to list to avoid conversion into binary data 
        y=y_data.tolist(),
        mode='markers',
        name='Daily Returns',
        marker=dict(color='rgba(0, 123, 255, 0.6)', size=8),
        hovertemplate=f"{name1}: %{{x:.4f}}<br>{name2}: %{{y:.4f}}<extra></extra>"
    ))
    
    # Add regression trendline
    try:
        X_reg = sm.add_constant(x_data)
        model = sm.OLS(y_data, X_reg).fit() # Ordinary Least Squares
        
        # Get statistics
        r_squared = model.rsquared
        
        # Create a smooth line for the trend
        x_range = np.linspace(x_data.min(), x_data.max(), 100)
        X_range_reg = sm.add_constant(x_range)
        predictions = model.get_prediction(X_range_reg)
        
        frame = predictions.summary_frame(alpha=0.05) # 95% confidence interval
        y_mean = frame['mean']
        y_upper = frame['mean_ci_upper']
        y_lower = frame['mean_ci_lower']
        
        fig.add_trace(go.Scatter(
            x=x_range.tolist() + x_range.tolist()[::-1],
            y=y_upper.tolist() + y_lower.tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence'
        ))
        
        # Plot the regression line
        fig.add_trace(go.Scatter(
            x=x_range.tolist(),
            y=y_mean.tolist(),
            mode='lines',
            name='OLS Trendline',
            line=dict(color='red', width=2)
        ))
        
        # Show the regression coefficients
        corr_coef = map2d.iloc[:, 0].corr(map2d.iloc[:, 1])
        fig.update_layout(title=f"Correlation: {name1} vs {name2} (Pearson r = {corr_coef:.3f} | R² = {r_squared:.3f})")
    except Exception as e:
        logger.error(f"Regression error: {e}")

    # Add the vertical and horizontal crosshair lines (at 0,0)
    fig.add_vline(x=0, line_dash="dash", line_color="grey", line_width=1)
    fig.add_hline(y=0, line_dash="dash", line_color="grey", line_width=1)

    fig.update_layout(
        template="plotly_white",
        xaxis_title=f"{name1} Returns",
        yaxis_title=f"{name2} Returns",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )

    return fig

def create_returns_distribution_chart(returns, student_t_params=None):
    """
    Distribution plot for returns.
    
    Input: DataFrame with one column of prices
    Output: Plotly figure (Histogram of returns)
    """
    logger.debug("create_returns_distribution_chart called")
    
    # Clean data in case it's not done before
    data = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    # If DataFrame, convert to Series by selecting first column
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    
    # If there's no data left, return a blank figure with a message
    if data.empty:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for returns", showarrow=False)
        return fig
        
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=data.tolist(),
        name='Return density',
        histnorm='probability density',
        marker=dict(
            color='#007BFF',
            line=dict(color='white', width=0.5) # Outline ensures visibility
        ),
        opacity=0.75,
        hovertemplate='Return: %{x:.2%}<br>Density: %{y}<extra></extra>'
    ))

    # Add vertical line for mean return
    mean_return = np.mean(data)
    fig.add_vline(x=mean_return, line_dash="dash", line_color="red", 
                  annotation_text=f"Mean: {mean_return:.2%}")

    # Add normal fit
    x_range = np.linspace(data.min(), data.max(), 100)
    y_pdf = stats.norm.pdf(x_range, loc=data.mean(), scale=data.std())

    fig.add_trace(go.Scatter(
                x=x_range.tolist(),
                y=y_pdf.tolist(),
                mode='lines',
                name='Normal dist.',
                hovertemplate=f"<br>Normal dist.: %{{y:.2f}}<extra></extra>"
            ))
            
    # Add a Student's t fit
    if student_t_params is None:
        student_t_params = stats.t.fit(data)

    student_t = stats.t.pdf(x_range,*student_t_params)
    
    fig.add_trace(go.Scatter(
                x=x_range.tolist(),
                y=student_t.tolist(),
                mode='lines',
                name='Students t dist.',
                hovertemplate=f"<br>Student's t dist.: %{{y:.2f}}<extra></extra>"
            ))

    fig.update_layout(
        xaxis_title="Daily returns",
        yaxis_title="Density",
        template="plotly_white",
        bargap=0.05,
        xaxis=dict(tickformat=".2%")
    )
    
    logger.debug("create_returns_distribution_chart out")
    
    return fig

def create_equity_curve_chart(equity_curve: pd.Series, title="Strategy equity curve") -> go.Figure:
    fig = go.Figure(go.Scatter(
        x=equity_curve.index.astype(str).tolist(),
        y=equity_curve.values.tolist(),
        mode='lines',
        name='Equity',
        hovertemplate='%{y:.2%}<extra></extra>'
    ))
    fig.update_layout(**_base_layout(title=title, yaxis_title='Cumulative return'))
    return fig

if __name__ == '__main__':

    simulated_stock_metrics = [
        {"Ticker": "StockA", "Months_Paid": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]},
        {"Ticker": "StockB", "Months_Paid": [0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]},
        {"Ticker": "StockC", "Months_Paid": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]},
        {"Ticker": "StockD", "Months_Paid": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]},
    ]
    #plot_file = create_monthly_dividends_plot(simulated_stock_metrics)
