
"""
1. parametros dia inicio de trade, delta leap, expiracion leap, delta expiracion pago, riesgo de la parte corta,la siguiente expiracion
2. Encontrar el trade
"""

import pandas as pd
import numpy as np
import requests

class estrategia_pagos:
    api = "43af86de-fd09-4fc4-b780-6a301d267cb2"
    def __init__(self):
        return None
    def buscar_leap(self, trade_date, ticker, dte, delta_leap):
        """
        """
        link = "https://api.orats.io/datav2/hist/strikes?token=" +  self.api + "&ticker=" + ticker + "&tradeDate=" + trade_date
        f = requests.get(link)
        f = f.json()
        db = pd.DataFrame(f['data'])
        # buscar el dte mas cercano a 365
        db['target_dte'] = np.abs(db['dte']-dte)
        db_leap = db[db['target_dte'] == db['target_dte'].min()]
        # buscar el delta mas cercano a delta_leap
        db_leap['target_delta'] = np.abs(db_leap['delta']-delta_leap)
        db_leap = db_leap[db_leap['target_delta'] == db_leap['target_delta'].min()]

        return db_leap
    
    def buscar_credit_spread(self, trade_date, ticker, dte_min, delta_credit, riesgo):
        link = "https://api.orats.io/datav2/hist/strikes?token=" +  self.api + "&ticker=" + ticker + "&tradeDate=" + trade_date
        f = requests.get(link)
        f = f.json()
        db = pd.DataFrame(f['data'])
        # buscar el dte mas cercano a 365
        db['target_dte'] = np.abs(db['dte']-dte_min)
        db_credit = db[db['target_dte'] == db['target_dte'].min()]
        # buscar el delta mas cercano a delta_leap
        db_credit['target_delta'] = np.abs(db_credit['delta']-delta_credit)
        db_credit = db_credit[db_credit['target_delta'] == db_credit['target_delta'].min()]
        # buscar el strike mas cercano riesgo
        db_debito = db[db['dte'] == db_credit['dte'].iloc[0]]
        db_debito = db_debito[db_debito['strike'] > db_credit['strike'].iloc[0]].sort_values(['strike']).iloc[[0]]
        return db_credit,db_debito
    
    def calcular_futuro(self, base_estudio,trade_date):
        trade_date = base_estudio['tradeDate'].iloc[0]
        strike = base_estudio['strike'].iloc[0]
        expirDate = base_estudio['expirDate'].iloc[0]
        ticker = base_estudio['ticker'].iloc[0]
        link = "https://api.orats.io/datav2/hist/strikes/options?token=" + self.api + "&ticker=" + ticker + "&expirDate=" + str(expirDate) + "&strike=" + str(strike)
        f = requests.get(link)
        f = f.json()
        db = pd.DataFrame(f['data'])
        db = db[pd.to_datetime(db['tradeDate']) >= pd.to_datetime(trade_date)]

        return db
    
    def compute_evolution(self, base, credito=False):
        if credito:
            base['ret'] = -(base['callValue'] - base['callValue'].iloc[0])
            base['ret_por'] = base['ret']/base['callValue'].iloc[0]
        else:
            base['ret'] = (base['callValue'] - base['callValue'].iloc[0])
            base['ret_por'] = base['ret']/base['callValue'].iloc[0]

        return base
    
    def calcular_put_a_delta(self, base, delta_thresh):
        """
        buscar la fecha de la primera vez que baja de 50 deltas
        """
        base = base[base['delta'] <= delta_thresh].head(1)
        return base
    
    def generar_pipeline(self, base_original_leap, base_original_credito, base_original_debito):
        fechas = pd.unique(base_original_leap['tradeDate'])
        loc = np.arange(len(fechas))
        # a partir del dia siguiente empiezo a calcular
        base_resultado = pd.DataFrame({'tradeDate': [fechas[0]],
                                       'expirDate_leap': [base_original_leap['expirDate'].iloc[0]],
                                       'dte_leap': [base_original_leap['dte'].iloc[0]],
                                       'strike_leap': [base_original_leap['strike'].iloc[0]],
                                       'stock_price': [base_original_leap['stockPrice'].iloc[0]],
                                       'call_precio': [base_original_leap['callValue'].iloc[0]],
                                       'delta_leap' : [base_original_leap['delta'].iloc[0]],
                                       'theta_leap' : [base_original_leap['theta'].iloc[0]],
                                       'vega_leap' : [base_original_leap['vega'].iloc[0]],
                                       'gamma_leap' : [base_original_leap['gamma'].iloc[0]],
                                       'iv_leap' : [base_original_leap['callMidIv'].iloc[0]],
                                       'call_credito': [base_original_credito['callValue'].iloc[0]],
                                       'call_debito': [base_original_debito['callValue'].iloc[0]],
                                       'strike_credito': [base_original_credito['strike'].iloc[0]],
                                       'strike_debito': [base_original_debito['strike'].iloc[0]],
                                       'id_leap':[base_original_leap['ticker'].iloc[0] + '_' + str(base_original_leap['strike'].iloc[0]) + '_' + str(base_original_leap['expirDate'].iloc[0])],
                                       'id_credito':[base_original_credito['ticker'].iloc[0] + '_' + str(base_original_credito['strike'].iloc[0]) + '_' + str(base_original_credito['expirDate'].iloc[0])],
                                       'id_debito':[base_original_debito['ticker'].iloc[0] + '_' + str(base_original_debito['strike'].iloc[0]) + '_' + str(base_original_debito['expirDate'].iloc[0])],

                                       })
        for fech in loc[1:2]:
            # si al final del dia siguiente el precio pasa el strike de venta o se paga mas del 80% del credito o el delta original baja de 0.50
            # Si ocurre que el precio de strike sobrepasa se cierra todo y se vuelve en esa fecha a generar una nueva Leap y spread
            # Si ocurre que se paga el 80% del credito se cierra el spread y se abre otro
            # si ocurre se incluye una put tal que el delta sea igual a 60
            print(fech)
            if base_original_leap['stockPrice'].iloc[fech] > base_original_credito['strike'].iloc[fech]:
                # se cierra todo y se abre una nueva leap y un spread
                a = 0
            else:

                base_tmp = pd.DataFrame({'tradeDate': [fechas[fech]],
                                               'expirDate_leap': [base_original_leap['expirDate'].iloc[fech]],
                                               'dte_leap': [base_original_leap['dte'].iloc[fech]],
                                               'strike_leap': [base_original_leap['strike'].iloc[fech]],
                                               'stock_price': [base_original_leap['stockPrice'].iloc[fech]],
                                               'call_precio': [base_original_leap['callValue'].iloc[fech]],
                                               'delta_leap' : [base_original_leap['delta'].iloc[fech]],
                                               'theta_leap' : [base_original_leap['theta'].iloc[fech]],
                                               'vega_leap' : [base_original_leap['vega'].iloc[fech]],
                                               'gamma_leap' : [base_original_leap['gamma'].iloc[fech]],
                                               'iv_leap' : [base_original_leap['callMidIv'].iloc[fech]],
                                               'call_credito': [base_original_credito['callValue'].iloc[fech]],
                                               'call_debito': [base_original_debito['callValue'].iloc[fech]],
                                               'strike_credito': [base_original_credito['strike'].iloc[fech]],
                                               'strike_debito': [base_original_debito['strike'].iloc[fech]],
                                               'id_leap':[base_original_leap['ticker'].iloc[fech] + '-' + str(base_original_leap['strike'].iloc[fech]) + '_' + str(base_original_leap['expirDate'].iloc[fech])],
                                               'id_credito':[base_original_credito['ticker'].iloc[fech] + '-' + str(base_original_credito['strike'].iloc[fech]) + '_' + str(base_original_credito['expirDate'].iloc[fech])],
                                               'id_debito':[base_original_debito['ticker'].iloc[fech] + '-' + str(base_original_debito['strike'].iloc[fech]) + '_' + str(base_original_debito['expirDate'].iloc[fech])],
                                       })
                base_resultado = pd.concat([base_resultado, base_tmp])




        return base_resultado



pag = estrategia_pagos()
wheel_leap = pag.buscar_leap(trade_date='2022-05-20', ticker='SPY', dte=365, delta_leap=0.65)
wheel_credit,wheel_debit = pag.buscar_credit_spread(trade_date='2022-05-20', ticker='SPY',  dte_min=7, delta_credit=0.20, riesgo=1)
base_leap = pag.calcular_futuro(base_estudio = wheel_leap,trade_date='2022-05-20')
wheel_credit = pag.calcular_futuro(base_estudio = wheel_credit,trade_date='2022-05-20')
wheel_debit = pag.calcular_futuro(base_estudio = wheel_debit,trade_date='2022-05-20')
base_leap = pag.compute_evolution(base = base_leap, credito = False)
base_credit = pag.compute_evolution(base = wheel_credit, credito = True)
base_debit = pag.compute_evolution(base = wheel_debit, credito = False)

print(pag.generar_pipeline(base_original_leap = base_leap, base_original_credito = base_credit, base_original_debito = base_debit))
# print(pag.calcular_put_a_delta(base = base_leap, delta_thresh = 0.50))


