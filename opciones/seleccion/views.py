from django.shortcuts import render
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans


def seleccion(request):
    #stock_picker = tickers_nifty50()
    #print(stock_picker)
    # va a llamar el html en el template
    # por el momento tomar todos los tickers
    stock_symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
    data = yf.download(stock_symbols, period="1d")
    current_prices = data['Close'].transpose().reset_index()
    date = current_prices.columns[-1]
    current_prices['date'] = date
    print("-----------")
    print(current_prices)
    current_prices.columns = ['ticker','price','date']
    current_prices = current_prices[['date','ticker','price']]
    current_prices = current_prices.to_dict('records')

    ### crear los pivotes ####

    ticker = 'CVS'
    start_date = '2010-05-01'
    end_date = '2025-03-02'
    window = 50
    threshold = 1.30
    n_clusters = 10


    df = yf.download(ticker, start=start_date, end=end_date)
    rolling_avg = df["Volume"].rolling(window=window).mean()
    relative_volume = df["Volume"] / rolling_avg
    highs = df["High"].values
    lows = df["Low"].values

    max_indices_all = argrelextrema(highs, np.greater_equal, order=window)[0]
    min_indices_all = argrelextrema(lows, np.less_equal, order=window)[0]
    high_pivots_filtered = []
    low_pivots_filtered = []

    high_pivots_filtered = []
    low_pivots_filtered = []

    for index in max_indices_all:
        if relative_volume.iloc[index][0] > threshold:
            high_pivots_filtered.append(index)

    for index in min_indices_all:
        if relative_volume.iloc[index][0] > threshold:
            low_pivots_filtered.append(index)
    
    high_pivots = df.iloc[high_pivots_filtered]
    low_pivots = df.iloc[low_pivots_filtered]
    print(high_pivots)
    print(low_pivots)

    prices = np.concatenate([df['High'].values.reshape(-1, 1), df['Low'].values.reshape(-1, 1)])
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)

    kmeans.fit(prices)
    cluster_centers = np.sort(kmeans.cluster_centers_.flatten())  # Sort cluster centers

    n_support = n_clusters // 2 # Integer division to handle odd clusters.
    support_levels = cluster_centers[:n_support]
    resistance_levels = cluster_centers[n_support:]

    print(support_levels)
    print(resistance_levels)

    return render(request, 'seleccion/base.html',{'current_prices':current_prices})

## features to select ticker,tradeDate,assetType,orFcst20d,orIvFcst20d,slope,slopeFcst,deriv,derivFcst,orHv1d,orHv5d,orHv10d,orHv20d,orHv60d,iv10d,iv20d,iv30d,iv60d,slopepctile,contango,wksNextErn,orHvXern5d,orHvXern10d,orHvXern20d,exErnIv10d,exErnIv20d,exErnIv30d,exErnIv60d
"""
1. Crear una tabla con las posiciones, sacado de la API, de forma local
2. Calcular las griegas y la volatilidad actual
"""