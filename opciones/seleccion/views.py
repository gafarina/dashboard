from django.shortcuts import render
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
import requests
import pandas as pd


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

def positions(request):
    link = "https://localhost:5000/v1/api/portfolio/U4264576/positions"
    g = requests.get(link, verify=False)
    g = g.json()
    posiciones = pd.DataFrame()
    posiciones_stk = pd.DataFrame()
    for i in g:
        print(i)
        print("-----------------------")
        if (i['assetClass'] == 'OPT') & (i['currency'] == 'USD'):
            pos = [x for x in i['contractDesc'].split('[')][1]
            exp = '20' + pos[6:12]
            exp1 = exp[0:4] + '-' + exp[4:6] + '-' + exp[6:8]
            ticker = pos[0:6].strip()
            tipo = pos[12:13]
            strike = float(str(int(pos[13:18])) + '.' + str(int(pos[18:20])))
            exp1 = pd.to_datetime(exp1, format='%Y-%m-%d')
            multiplicador = pos[-4:-1]
            posiciones_tmp = pd.DataFrame({ 'ticker': [ticker],
                                            'exp': exp1,
                                            'strike': strike,
                                            'call_put': tipo,
                                            'position': i['position'],
                                            'mktPrice': i['mktPrice'],
                                            'multiplicador': int(multiplicador),
                                            'mktValue':i['mktValue'],
                                            'currency': i['currency'],
                                            'avgPrice': i['avgPrice'],
                                            'unrealizedPnl':i['unrealizedPnl'],
                                            'conid':i['conid'],
                                            'assetClass':i['assetClass']})
            posiciones = pd.concat([posiciones, posiciones_tmp])            
        elif i['assetClass'] == 'STK':
            pos = [x for x in i['contractDesc'].split(' ') if x != '']
            posiciones_tmp_stk = pd.DataFrame({ 'ticker': [pos[0].strip()],
                                            'exp': np.nan,
                                            'strike': np.nan,
                                            'call_put': np.nan,
                                            'position': i['position'],
                                            'mktPrice': i['mktPrice'],
                                            'multiplicador': 1,
                                            'mktValue':i['mktValue'],
                                            'currency': i['currency'],
                                            'avgPrice': i['avgPrice'],
                                            'unrealizedPnl':i['unrealizedPnl'],
                                            'conid':i['conid'],
                                            'assetClass':i['assetClass']})
            posiciones_stk = pd.concat([posiciones_stk, posiciones_tmp_stk])
    posiciones_final = pd.concat([posiciones, posiciones_stk])
    posiciones_final['exp'] = posiciones_final['exp'].fillna('1982-05-13')
    posiciones_final_gb = posiciones_final.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    posiciones_final_gb['unrealizedPnl'] = np.round(posiciones_final_gb['unrealizedPnl'],0)

    # get delta
    posiciones_final['conid'] = posiciones_final['conid'].astype('str')
    contratos = ','.join(posiciones_final['conid'])
    print(contratos)
    
    link_delta = "https://localhost:5000/v1/api/iserver/marketdata/snapshot?conids=" + str(contratos) + "&fields=7308,7309,7310,7311,31,7633,6457"
    print(link_delta)
    g = requests.get(link_delta, verify=False)
    print(g)
    g = g.json()
    base_opciones = pd.DataFrame(g)
    base_opciones = base_opciones[['conid','31','7308','7309','7310','7311','7633','6457']]
    base_opciones.columns = ['conid','precio','delta','gamma','theta','vega','iv','conid_sub']
    print(base_opciones)
    # calcular los precios de los subyacentes
    subyacentes = list(pd.unique(base_opciones['conid_sub'].astype('str')))
    subyacentes = ','.join(subyacentes)
    link_subyacentes = "https://localhost:5000/v1/api/iserver/marketdata/snapshot?conids=" + str(subyacentes) + "&fields=55,31,7287,7655"
    g = requests.get(link_subyacentes, verify=False)
    print(g)
    g = g.json()
    base_subs = pd.DataFrame(g)
    base_subs = base_subs[['conid','55','31','7287']]
    base_subs.columns = ['conid_sub','ticker','precio_sub','dividendo']

    base_opciones['conid'] = base_opciones['conid'].astype('str')
    posiciones_final['conid'] = posiciones_final['conid'].astype('str')

    posiciones_final = pd.merge(posiciones_final, base_opciones, on=['conid'], how='left')
    # juntar las griegas y volatilidad
    base_subs['conid_sub'] = base_subs['conid_sub'].astype('str')
    posiciones_final = pd.merge(posiciones_final, base_subs, on=['conid_sub'], how='left')

    posiciones_final['delta'] = posiciones_final['delta'].fillna(0)
    posiciones_final['position'] = posiciones_final['position'].astype('float')
    posiciones_final['delta'] = posiciones_final['delta'].astype('float')
    posiciones_final['gamma'] = posiciones_final['gamma'].astype('float')
    posiciones_final['theta'] = posiciones_final['theta'].astype('float')
    posiciones_final['vega'] = posiciones_final['vega'].astype('float')

    posiciones_final['delta_pos'] = posiciones_final['delta']*posiciones_final['position']*100
    posiciones_final['gamma_pos'] = posiciones_final['gamma']*posiciones_final['position']*100
    posiciones_final['theta_pos'] = posiciones_final['theta']*posiciones_final['position']*100
    posiciones_final['vega_pos'] = posiciones_final['vega']*posiciones_final['position']*100

    posiciones_final = posiciones_final.rename(columns={'ticker_x':'ticker'})

    posiciones_final_gb = posiciones_final.groupby(['ticker'])[['unrealizedPnl','delta_pos', 'gamma_pos', 'theta_pos', 'vega_pos']].sum().reset_index()
    posiciones_final_gb['unrealizedPnl'] = np.round(posiciones_final_gb['unrealizedPnl'],0)
    posiciones_final_gb['delta_pos'] = np.round(posiciones_final_gb['delta_pos'],2)
    posiciones_final_gb['gamma_pos'] = np.round(posiciones_final_gb['gamma_pos'],2)
    posiciones_final_gb['theta_pos'] = np.round(posiciones_final_gb['theta_pos'],2)
    posiciones_final_gb['vega_pos'] = np.round(posiciones_final_gb['vega_pos'],2)

    posiciones_final['unrealizedPnl'] = np.round(posiciones_final['unrealizedPnl'],0)
    posiciones_final['delta_pos'] = np.round(posiciones_final['delta_pos'],2)
    posiciones_final['gamma_pos'] = np.round(posiciones_final['gamma_pos'],2)
    posiciones_final['theta_pos'] = np.round(posiciones_final['theta_pos'],2)
    posiciones_final['vega_pos'] = np.round(posiciones_final['vega_pos'],2)

    posiciones_final['strike'] = posiciones_final['strike'].astype('float')
    posiciones_final['precio_sub'] = posiciones_final['precio_sub'].str.replace('C','')
    posiciones_final['precio_sub'] = posiciones_final['precio_sub'].astype('float')

    posiciones_final['precio_dist'] = ((posiciones_final['precio_sub']/posiciones_final['strike'])-1)*100
    posiciones_final['precio_dist'] = np.round(posiciones_final['precio_dist'],3)
   
    # pl short position
    posiciones_final_short = posiciones_final[posiciones_final['position'] < 0]
    posiciones_final_short_gb = posiciones_final_short.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    posiciones_final_short_gb.columns = ['ticker', 'unrealizedPnl_short']
    posiciones_final_gb = pd.merge(posiciones_final_gb, posiciones_final_short_gb, on=['ticker'], how='left')
    posiciones_final_gb['unrealizedPnl_short'] = np.round(posiciones_final_gb['unrealizedPnl_short'],0)

    # pl short position
    posiciones_final_long = posiciones_final[posiciones_final['position'] >= 0]
    posiciones_final_long_gb = posiciones_final_long.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    posiciones_final_long_gb.columns = ['ticker', 'unrealizedPnl_long']
    posiciones_final_gb = pd.merge(posiciones_final_gb, posiciones_final_long_gb, on=['ticker'], how='left')
    posiciones_final_gb['unrealizedPnl_long'] = np.round(posiciones_final_gb['unrealizedPnl_long'],0)


    
    return render(request, 'seleccion/base.html',{'current_prices':posiciones_final_gb.to_dict('records'), 'posiciones_final':posiciones_final.to_dict('records')})


## features to select ticker,tradeDate,assetType,orFcst20d,orIvFcst20d,slope,slopeFcst,deriv,derivFcst,orHv1d,orHv5d,orHv10d,orHv20d,orHv60d,iv10d,iv20d,iv30d,iv60d,slopepctile,contango,wksNextErn,orHvXern5d,orHvXern10d,orHvXern20d,exErnIv10d,exErnIv20d,exErnIv30d,exErnIv60d
"""
1. Crear una tabla con las posiciones, sacado de la API, de forma local
2. Calcular las griegas y la volatilidad actual
"""