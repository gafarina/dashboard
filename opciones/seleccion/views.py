from django.shortcuts import render
import yfinance as yf
import numpy as np
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
import requests
import pandas as pd
from .clase_opciones import opciones


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
    ## calcular las posiciones de mi cartera
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

    # los valores de los stocks estan en nan y los strikes dejarlos en cero
    posiciones_final['exp'] = posiciones_final['exp'].fillna('1982-05-13')
    posiciones_final.loc[posiciones_final['call_put'].isnull(), 'call_put'] = 'S'
    posiciones_final.loc[posiciones_final['strike'].isnull(), 'strike'] = 0

    # necesito calcular el precio del subyacente y las griegas para cada contrato
    posiciones_final['conid'] = posiciones_final['conid'].astype('str')
    contratos = ','.join(posiciones_final['conid'])
    link_delta = "https://localhost:5000/v1/api/iserver/marketdata/snapshot?conids=" + str(contratos) + "&fields=7308,7309,7310,7311,31,7633,6457"
    g = requests.get(link_delta, verify=False)
    g = g.json()
    print(g)
    base_opciones = pd.DataFrame(g)
    base_opciones = base_opciones[['conid','31','7308','7309','7310','7311','7633','6457']]
    base_opciones.columns = ['conid','precio','delta','gamma','theta','vega','iv','conid_sub']
    subyacentes = list(pd.unique(base_opciones['conid_sub'].astype('str')))
    subyacentes = ','.join(subyacentes)
    link_subyacentes = "https://localhost:5000/v1/api/iserver/marketdata/snapshot?conids=" + str(subyacentes) + "&fields=55,31,7287,7655"
    g = requests.get(link_subyacentes, verify=False)
    g = g.json()
    base_subs = pd.DataFrame(g)
    base_subs = base_subs[['conid','55','31','7287']]
    base_subs.columns = ['conid_sub','ticker','precio_sub','dividendo']

    base_opciones['conid'] = base_opciones['conid'].astype('str')
    posiciones_final['conid'] = posiciones_final['conid'].astype('str')
    base_subs['conid_sub'] = base_subs['conid_sub'].astype('str')

    posiciones_final = pd.merge(posiciones_final, base_opciones, on=['conid'], how='left')
    posiciones_final = pd.merge(posiciones_final, base_subs, on=['conid_sub'], how='left')
    print("test")
    print(posiciones_final[posiciones_final['ticker_x'] == 'SPY'])

    # formateo iv
    posiciones_final['iv'] = posiciones_final['iv'].str.replace("%",'').astype('float')/100
    # calculo del iv usando BS
    # calculo del ultimo precio de las opciones
    posiciones_final['precio'] = posiciones_final['precio'].str.replace("C",'').astype('float')
    posiciones_final['dividendo'] = posiciones_final['dividendo'].str.replace("%",'').astype('float')/100
    posiciones_final['dividendo'] = posiciones_final['dividendo'].fillna(0)
    posiciones_final['interest_rate'] = 0.045
    posiciones_final['strike'] = posiciones_final['strike'].astype('float')
    posiciones_final['precio_sub'] = posiciones_final['precio_sub'].str.replace("C",'').astype('float')
    #posiciones_final['precio_sub'] = posiciones_final['precio_sub'].astype('float')
    # con el ultimo precio puedo obtener la ultima volatilidad
    # time to maturiry
    posiciones_final['TimeToMaturity'] = (pd.to_datetime(posiciones_final['exp'])-pd.to_datetime('today')).dt.total_seconds()/60 + 60*24
    posiciones_final['TimeToMaturity'] = posiciones_final['TimeToMaturity']/(365*24*60)
    posiciones_final['position'] = posiciones_final['position'].astype('float')

    op = opciones()
    #posiciones_final['iv_model'] = posiciones_final.apply(lambda row: op.implied_volatility_american_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['precio']), axis=1)
    #posiciones_final['delta_model'] = ''
    #posiciones_final['iv_model'] = posiciones_final.apply(lambda row: op.implied_volatility_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['precio'], row['call_put'], initial_guess=0.2),axis=1)
    
    posiciones_final_call = posiciones_final[posiciones_final['call_put'] == 'C']
    posiciones_final_put = posiciones_final[posiciones_final['call_put'] == 'P']
    # print(posiciones_final_call)
    
    # print("Test_call")
    # print(posiciones_final_call[['precio_sub','strike','TimeToMaturity','interest_rate','dividendo','precio','call_put']])
    
    
    posiciones_final_call['iv_model'] = posiciones_final_call.apply(lambda row: op.implied_volatility_with_dividends(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['precio'], row['call_put']),axis=1)
    posiciones_final_put['iv_model'] = posiciones_final_put.apply(lambda row: op.implied_volatility_with_dividends(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['precio'], row['call_put']),axis=1)
    
    posiciones_final_call.loc[posiciones_final_call['iv_model'] > 2, 'iv_model'] = 1
    posiciones_final_put.loc[posiciones_final_put['iv_model'] > 2, 'iv_model'] = 1
    posiciones_final_call.loc[posiciones_final_call['iv_model'].isnull(), 'iv_model'] = 1
    posiciones_final_put.loc[posiciones_final_put['iv_model'].isnull(), 'iv_model'] = 1

    posiciones_final_call.loc[posiciones_final_call['iv'].isnull(), 'iv'] = posiciones_final_call.loc[posiciones_final_call['iv'].isnull(), 'iv_model']
    posiciones_final_put.loc[posiciones_final_put['iv'].isnull(), 'iv'] = posiciones_final_put.loc[posiciones_final_put['iv'].isnull(), 'iv_model']


 

    posiciones_final_call['delta_model'] = posiciones_final_call.apply(lambda row: op.european_call_delta_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)
    posiciones_final_put['delta_model'] = posiciones_final_put.apply(lambda row: op.european_put_delta_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)

    posiciones_final_call['gamma_model'] = posiciones_final_call.apply(lambda row: op.gamma_call_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)
    posiciones_final_put['gamma_model'] = posiciones_final_put.apply(lambda row: op.gamma_call_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)

    posiciones_final_call['theta_model'] = posiciones_final_call.apply(lambda row: op.theta_call_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)
    posiciones_final_put['theta_model'] = posiciones_final_put.apply(lambda row: op.theta_put_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)

    posiciones_final_call['vega_model'] = posiciones_final_call.apply(lambda row: op.vega_european_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)
    posiciones_final_put['vega_model'] = posiciones_final_put.apply(lambda row: op.vega_european_dividend(row['precio_sub'], row['strike'], row['TimeToMaturity'],row['interest_rate'], row['dividendo'], row['iv']), axis=1)

    posiciones_final_call['theta_model'] = posiciones_final_call['theta_model']/365
    posiciones_final_put['theta_model'] = posiciones_final_put['theta_model']/365
    posiciones_final_put['vega_model'] = posiciones_final_put['vega_model']/100
    posiciones_final_call['vega_model'] = posiciones_final_call['vega_model']/100
    
    posiciones_final_call.loc[posiciones_final_call['delta'].isnull(), 'delta'] = posiciones_final_call.loc[posiciones_final_call['delta'].isnull(), 'delta_model']
    posiciones_final_put.loc[posiciones_final_put['delta'].isnull(), 'delta'] = posiciones_final_put.loc[posiciones_final_put['iv'].isnull(), 'delta_model']

    posiciones_final_call.loc[posiciones_final_call['gamma'].isnull(), 'gamma'] = posiciones_final_call.loc[posiciones_final_call['gamma'].isnull(), 'gamma_model']
    posiciones_final_put.loc[posiciones_final_put['gamma'].isnull(), 'gamma'] = posiciones_final_put.loc[posiciones_final_put['gamma'].isnull(), 'gamma_model']

    posiciones_final_call.loc[posiciones_final_call['vega'].isnull(), 'vega'] = posiciones_final_call.loc[posiciones_final_call['vega'].isnull(), 'vega_model']
    posiciones_final_put.loc[posiciones_final_put['vega'].isnull(), 'vega'] = posiciones_final_put.loc[posiciones_final_put['vega'].isnull(), 'vega_model']

    posiciones_final_call.loc[posiciones_final_call['theta'].isnull(), 'theta'] = posiciones_final_call.loc[posiciones_final_call['theta'].isnull(), 'theta_model']
    posiciones_final_put.loc[posiciones_final_put['theta'].isnull(), 'theta'] = posiciones_final_put.loc[posiciones_final_put['theta'].isnull(), 'theta_model']

    base_opciones = pd.concat([posiciones_final_call,posiciones_final_put])
    base_opciones = base_opciones[['ticker_x','call_put','position','exp','strike','delta','gamma','vega','theta','iv']]
    base_opciones.columns = ['ticker_x','call_put','position','exp','strike','delta_final','gamma_final','vega_final','theta_final','iv_final']
    
    posiciones_final = pd.merge(posiciones_final, base_opciones, on=['ticker_x','call_put','position','exp','strike'], how='left')
    posiciones_final.loc[posiciones_final['call_put'] == 'S','delta_final'] = posiciones_final.loc[posiciones_final['call_put'] == 'S','position']/100
    posiciones_final.loc[posiciones_final['call_put'] == 'S','gamma_final'] = 0
    posiciones_final.loc[posiciones_final['call_put'] == 'S','vega_final'] = 0
    posiciones_final.loc[posiciones_final['call_put'] == 'S','theta_final'] = 0
    posiciones_final.loc[posiciones_final['call_put'] == 'S','iv_final'] = 0
    # print(posiciones_final)
    # print(posiciones_final.columns)

    posiciones_final['delta_final'] = posiciones_final['delta_final'].astype('float')
    posiciones_final['theta_final'] = posiciones_final['theta_final'].astype('float')
    posiciones_final['vega_final'] = posiciones_final['vega_final'].astype('float')
    posiciones_final['gamma_final'] = posiciones_final['gamma_final'].astype('float')
    posiciones_final['position'] = posiciones_final['position'].astype('float')

    print(posiciones_final)
    posiciones_final.loc[posiciones_final['call_put'] == 'S','delta_final'] = posiciones_final.loc[posiciones_final['call_put'] == 'S','position']/100
    posiciones_final['delta_pos'] = posiciones_final['delta_final']*posiciones_final['position']*100
    posiciones_final['gamma_pos'] = posiciones_final['gamma_final']*posiciones_final['position']*100
    posiciones_final['theta_pos'] = posiciones_final['theta_final']*posiciones_final['position']*100
    posiciones_final['vega_pos'] = posiciones_final['vega_final']*posiciones_final['position']*100
    posiciones_final.loc[posiciones_final['call_put'] == 'S','delta_pos'] = posiciones_final.loc[posiciones_final['call_put'] == 'S','position']/100

    posiciones_final = posiciones_final.rename(columns = {'ticker_x':'ticker'})
    #print(posiciones_final)

    posiciones_final_gb = posiciones_final.groupby(['ticker'])[['unrealizedPnl','delta_pos', 'gamma_pos', 'theta_pos', 'vega_pos']].sum().reset_index()
    posiciones_final_gb['unrealizedPnl'] = np.round(posiciones_final_gb['unrealizedPnl'],0)
    posiciones_final_gb['delta_pos'] = np.round(posiciones_final_gb['delta_pos'],2)
    posiciones_final_gb['gamma_pos'] = np.round(posiciones_final_gb['gamma_pos'],2)
    posiciones_final_gb['theta_pos'] = np.round(posiciones_final_gb['theta_pos'],2)
    posiciones_final_gb['vega_pos'] = np.round(posiciones_final_gb['vega_pos'],2)

    # # pl short position
    posiciones_final_short = posiciones_final[posiciones_final['position'] < 0]
    posiciones_final_short_gb = posiciones_final_short.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    posiciones_final_short_gb.columns = ['ticker', 'unrealizedPnl_short']
    posiciones_final_gb = pd.merge(posiciones_final_gb, posiciones_final_short_gb, on=['ticker'], how='left')
    posiciones_final_gb['unrealizedPnl_short'] = np.round(posiciones_final_gb['unrealizedPnl_short'],0)

    # # pl short position
    posiciones_final_long = posiciones_final[posiciones_final['position'] >= 0]
    posiciones_final_long_gb = posiciones_final_long.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    posiciones_final_long_gb.columns = ['ticker', 'unrealizedPnl_long']
    posiciones_final_gb = pd.merge(posiciones_final_gb, posiciones_final_long_gb, on=['ticker'], how='left')
    posiciones_final_gb['unrealizedPnl_long'] = np.round(posiciones_final_gb['unrealizedPnl_long'],0)

    ### take trades
    base_trades = pd.read_csv("C:\\Users\\gasto\\OneDrive\\Opciones_Django\\opciones\\seleccion\\trades\\trades_13_03_2025.csv")
    base_trades = base_trades[['Symbol', 'Description', 'UnderlyingSymbol', 'TradeDate', 'Quantity', 'Buy/Sell', 'ReportDate','DateTime','IBCommission','NetCash','MtmPnl','ClosePrice','CostBasis']]
    #print(base_trades[base_trades['UnderlyingSymbol'] == 'UBER'])
    ### tomar todos los contratos que tengan un numero par de apariciones
    trades_cerrados = base_trades.groupby(['Description']).agg({'Symbol':len, 'Quantity': np.sum, 'UnderlyingSymbol':max}).reset_index()
    trades_cerrados = trades_cerrados[trades_cerrados['Quantity'] == 0]
    contratos_cerrados = pd.unique(trades_cerrados['Description'])
    base_trades = base_trades[base_trades['Description'].isin(contratos_cerrados)]
    trades_cerrados_acumulados = base_trades.groupby(['UnderlyingSymbol'])[['NetCash','IBCommission']].sum().reset_index()
    trades_cerrados_acumulados.columns = ['ticker','p_l_realized','commision']
    tickers_activos = pd.unique(posiciones_final_gb['ticker'])
    trades_cerrados_acumulados['cobertura'] = 0
    trades_cerrados_acumulados['cobertura_ib_commison'] = 0
    trades_cerrados_acumulados['cobertura'] = trades_cerrados_acumulados[~trades_cerrados_acumulados['ticker'].isin(tickers_activos)]['p_l_realized'].sum()
    trades_cerrados_acumulados['cobertura_ib_commison'] = trades_cerrados_acumulados[~trades_cerrados_acumulados['ticker'].isin(tickers_activos)]['commision'].sum()
    
    posiciones_final_gb = pd.merge(posiciones_final_gb, trades_cerrados_acumulados, on=['ticker'], how='left')
    posiciones_final_gb['p_l'] = posiciones_final_gb['p_l_realized'] + posiciones_final_gb['commision'] + posiciones_final_gb['unrealizedPnl']
    posiciones_final_gb['p_l'] = posiciones_final_gb['p_l'].fillna(0)
    posiciones_final_gb['cobertura_ib_commison'] = posiciones_final_gb['cobertura_ib_commison'].fillna(0)
    posiciones_final_gb['cobertura'] = posiciones_final_gb['cobertura'].fillna(0)
    posiciones_final_gb['commision'] = posiciones_final_gb['commision'].fillna(0)
    posiciones_final_gb['p_l_realized'] = posiciones_final_gb['p_l_realized'].fillna(0)
    posiciones_final_gb['p_l_realized_total'] = posiciones_final_gb['p_l_realized'].sum()
    posiciones_final_gb['p_l_total'] = posiciones_final_gb['p_l'].sum() + posiciones_final_gb['cobertura'].max() + posiciones_final_gb['cobertura_ib_commison'].min()
    posiciones_final_gb['cobertura_ib_commison_total'] = posiciones_final_gb['commision'].sum() + posiciones_final_gb['cobertura_ib_commison'].min()
    posiciones_final_gb['p_l_realized'] = np.round(posiciones_final_gb['p_l_realized'],2)
    posiciones_final_gb['p_l_realized_total'] = np.round(posiciones_final_gb['p_l_realized_total'],2)
    posiciones_final_gb['commision'] = np.round(posiciones_final_gb['commision'],2)
    posiciones_final_gb['cobertura'] = np.round(posiciones_final_gb['cobertura'],2)
    posiciones_final_gb['cobertura_ib_commison'] = np.round(posiciones_final_gb['cobertura_ib_commison'],2)
    posiciones_final_gb['p_l_total'] = np.round(posiciones_final_gb['p_l_total'],2)
    posiciones_final_gb['p_l'] = np.round(posiciones_final_gb['p_l'],2)

    #print(posiciones_final_gb.columns)
    #symbolos_vigentes = list(pd.unique(base_trades['ticker']))

    # # get greeks
    # posiciones_final['conid'] = posiciones_final['conid'].astype('str')
    # contratos = ','.join(posiciones_final['conid'])
    
    # link_delta = "https://localhost:5000/v1/api/iserver/marketdata/snapshot?conids=" + str(contratos) + "&fields=7308,7309,7310,7311,31,7633,6457"
    # g = requests.get(link_delta, verify=False)
    # print(g)
    # g = g.json()
    # base_opciones = pd.DataFrame(g)
    # base_opciones = base_opciones[['conid','31','7308','7309','7310','7311','7633','6457']]
    # base_opciones.columns = ['conid','precio','delta','gamma','theta','vega','iv','conid_sub']
    # # calcular los precios de los subyacentes
    # subyacentes = list(pd.unique(base_opciones['conid_sub'].astype('str')))
    # subyacentes = ','.join(subyacentes)
    # link_subyacentes = "https://localhost:5000/v1/api/iserver/marketdata/snapshot?conids=" + str(subyacentes) + "&fields=55,31,7287,7655"
    # g = requests.get(link_subyacentes, verify=False)
    # print(g)
    # g = g.json()
    # base_subs = pd.DataFrame(g)
    # base_subs = base_subs[['conid','55','31','7287']]
    # base_subs.columns = ['conid_sub','ticker','precio_sub','dividendo']

    # base_opciones['conid'] = base_opciones['conid'].astype('str')
    # posiciones_final['conid'] = posiciones_final['conid'].astype('str')

    # posiciones_final = pd.merge(posiciones_final, base_opciones, on=['conid'], how='left')
    # # juntar las griegas y volatilidad
    # base_subs['conid_sub'] = base_subs['conid_sub'].astype('str')
    # posiciones_final = pd.merge(posiciones_final, base_subs, on=['conid_sub'], how='left')

    # #posiciones_final['delta'] = posiciones_final['delta'].fillna(0)
    # posiciones_final['position'] = posiciones_final['position'].astype('float')
    # posiciones_final['delta'] = posiciones_final['delta'].astype('float')
    # posiciones_final['gamma'] = posiciones_final['gamma'].astype('float')
    # posiciones_final['theta'] = posiciones_final['theta'].astype('float')
    # posiciones_final['vega'] = posiciones_final['vega'].astype('float')


    # posiciones_final['delta_pos'] = posiciones_final['delta']*posiciones_final['position']*100
    # posiciones_final['gamma_pos'] = posiciones_final['gamma']*posiciones_final['position']*100
    # posiciones_final['theta_pos'] = posiciones_final['theta']*posiciones_final['position']*100
    # posiciones_final['vega_pos'] = posiciones_final['vega']*posiciones_final['position']*100

    # posiciones_final = posiciones_final.rename(columns={'ticker_x':'ticker'})

    # posiciones_final_gb = posiciones_final.groupby(['ticker'])[['unrealizedPnl','delta_pos', 'gamma_pos', 'theta_pos', 'vega_pos']].sum().reset_index()
    # posiciones_final_gb['unrealizedPnl'] = np.round(posiciones_final_gb['unrealizedPnl'],0)
    # posiciones_final_gb['delta_pos'] = np.round(posiciones_final_gb['delta_pos'],2)
    # posiciones_final_gb['gamma_pos'] = np.round(posiciones_final_gb['gamma_pos'],2)
    # posiciones_final_gb['theta_pos'] = np.round(posiciones_final_gb['theta_pos'],2)
    # posiciones_final_gb['vega_pos'] = np.round(posiciones_final_gb['vega_pos'],2)

    # posiciones_final['unrealizedPnl'] = np.round(posiciones_final['unrealizedPnl'],0)
    # posiciones_final['delta_pos'] = np.round(posiciones_final['delta_pos'],2)
    # posiciones_final['gamma_pos'] = np.round(posiciones_final['gamma_pos'],2)
    # posiciones_final['theta_pos'] = np.round(posiciones_final['theta_pos'],2)
    # posiciones_final['vega_pos'] = np.round(posiciones_final['vega_pos'],2)

    # posiciones_final['strike'] = posiciones_final['strike'].astype('float')
    # posiciones_final['precio_sub'] = posiciones_final['precio_sub'].str.replace('C','')
    # posiciones_final['precio_sub'] = posiciones_final['precio_sub'].astype('float')

    # posiciones_final['precio_dist'] = ((posiciones_final['precio_sub']/posiciones_final['strike'])-1)*100
    # posiciones_final['precio_dist'] = np.round(posiciones_final['precio_dist'],3)
   
    # # pl short position
    # posiciones_final_short = posiciones_final[posiciones_final['position'] < 0]
    # posiciones_final_short_gb = posiciones_final_short.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    # posiciones_final_short_gb.columns = ['ticker', 'unrealizedPnl_short']
    # posiciones_final_gb = pd.merge(posiciones_final_gb, posiciones_final_short_gb, on=['ticker'], how='left')
    # posiciones_final_gb['unrealizedPnl_short'] = np.round(posiciones_final_gb['unrealizedPnl_short'],0)

    # # pl short position
    # posiciones_final_long = posiciones_final[posiciones_final['position'] >= 0]
    # posiciones_final_long_gb = posiciones_final_long.groupby(['ticker'])[['unrealizedPnl']].sum().reset_index()
    # posiciones_final_long_gb.columns = ['ticker', 'unrealizedPnl_long']
    # posiciones_final_gb = pd.merge(posiciones_final_gb, posiciones_final_long_gb, on=['ticker'], how='left')
    # posiciones_final_gb['unrealizedPnl_long'] = np.round(posiciones_final_gb['unrealizedPnl_long'],0)


    
    return render(request, 'seleccion/base.html',{'current_prices':posiciones_final_gb.to_dict('records'), 'posiciones_final':posiciones_final.to_dict('records')})


# def market_analisis(request):
#     """
#     1. buscar los stocks que tienen las mejores estrategias
#     2. Buscar los universos con mas volumen
#     1. calcular los que esten en mayor backwardation 30 y 20 dias, 60 y 30 dias, sacar un promedio
#     2. Calcular el slope a 30 y 60 dias, calls, dividido por puts
#     3. calculo de volatilidad a 30 y 60 dias 
#     4. ranking de los puntos 1, 2, y 3
#     5. porcentaje de cambio en el precio de la ultima semana y ranking de la ultima semana y ultimos 20 dias
#     6. Calcular el RSI y el promedio
#     6. calcular el promedio de todos los puntos anteriores y generar los pesos en base a las estrategias
#     7. momentum 

#     calendarios OTM : contango mas fuerte, volatilidad, ranking volatilidad, slope ....
#     Credit Spread: volitilidad, slope
#     short puts, calls: volatilidad, caida, 
#     diagonals: contango, volatilidad, ranking voilatilidad, slope


#     Plan General
#     0. Tener los tickers con mas volumen al ultimo dia
#     1. Generar una base de datos con los datos historicos, de los datos generales(empezar con un 1 año)
#     2. Si el dia de consulta es habil y no es el ultimo valor de la base, se rellenan los datos historicos hasta el ultimo dia habil
#     3. Si el dia habil de la consulta tiene valores y es distinto a la ultima fecha se va incluyendo
#     4. Para cada actualizacion se van bajando los datos del ultimo dia actualizados
#     5. Calcular los indicadores de volatilidad
#     6. hacer una base de datos historicos de precios de los tickers por un año (yahoo)
#     7. Con la misma logica de actualizacion que las volatilidades hacer yahoo
#     8. calcular el RSI
#     9. los retornos y las medias moviles

#     """
#     posiciones_final_gb = pd.DataFrame()
#     return render(request, 'seleccion/ticker.html',{'current_prices':posiciones_final_gb.to_dict('records')})



## features to select ticker,tradeDate,assetType,orFcst20d,orIvFcst20d,slope,slopeFcst,deriv,derivFcst,orHv1d,orHv5d,orHv10d,orHv20d,orHv60d,iv10d,iv20d,iv30d,iv60d,slopepctile,contango,wksNextErn,orHvXern5d,orHvXern10d,orHvXern20d,exErnIv10d,exErnIv20d,exErnIv30d,exErnIv60d
"""
1. Crear una tabla con las posiciones, sacado de la API, de forma local
2. Calcular las griegas y la volatilidad actual
"""