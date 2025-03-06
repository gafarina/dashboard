import pandas as pd
import requests
import pandas_market_calendars as mcal
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import numpy as np
import math
class opciones:
    """Class of the computation of the ranking of best strategy"""

    # Class attributes
    api = "43af86de-fd09-4fc4-b780-6a301d267cb2"

    def get_core_data_historical(self, list_dates):
        """
            calcula los datos cores para las variables anteriores descritas y para cada fecha se van concatenando, se van guardando en un pickle
            list_date: lista de fechas que voy a ir concatenando
        """
        df_total = pd.DataFrame()
        nzdq = mcal.get_calendar('NASDAQ')
        trading_days = nzdq.valid_days(start_date=list_dates[0], end_date=str(datetime.date.today()))
        trading_days = [str(x)[0:10] for x in trading_days]
        last_day = trading_days[-2]
        print(last_day)
        ## get the tickers with more volume in the last day
        try:
            link = "https://api.orats.io/datav2/hist/cores?token=" + self.api + "&tradeDate=" + str(last_day)
            f = requests.get(link)
            f = f.json()
            db = pd.DataFrame(f['data'])
            # tomar solo los que el cVolu + pVolu y cOi + pOi sea mayor que el cuantil 95
            db['option_volume'] = db['cVolu'] + db['pVolu']
            db['option_oi'] = db['cOi'] + db['pOi']
            db = db[(db['option_volume'] >= db['option_volume'].quantile(0.70)) & (db['option_volume'] >= db['option_oi'].quantile(0.70))]
            df_total = pd.concat([df_total, db])
            tickers = list(pd.unique(df_total['ticker']))
        except:
            tickers = []
        print("Los tickers en el universo son {}".format(str(tickers)))
        for date in trading_days:
            print("Se esta procesando {} ".format(str(date)))
            try:
                link = "https://api.orats.io/datav2/hist/cores?token=" + self.api + "&tradeDate=" + str(date)
                f = requests.get(link)
                f = f.json()
                db = pd.DataFrame(f['data'])
                db = db[db['ticker'].isin(tickers)]
                df_total = pd.concat([df_total, db])
                df_total.to_pickle("C:\\Users\\gasto\\OneDrive\\base_datos_proyecto.pkl")
            except:
                print("existio un problema con la fecha {}".format(str(date)))
                db = pd.DataFrame()
                df_total = pd.concat([df_total, db])
        #print(df_total)
        return None

    def read_train_data(self, path):
        """
            Filtra los  features para clasificar
            path : ruta del archivo de clasificacion
        """
        base = pd.read_pickle(path)
        base = base[['ticker','tradeDate','pxAtmIv','mktCap','orFcst20d','orIvFcst20d','orFcstInf','orIvXern20d', 'orIvXernInf','volOfVol', 'volOfIvol',
                     'slope','slopeInf','slopeFcst','slopeFcstInf','deriv','derivInf','derivFcst','derivFcstInf','mktWidthVol','mktWidthVolInf',
                     'orHv1d','orHv5d','orHv10d','orHv20d','orHv60d','orHv90d','orHv120d','orHv252d','orHv500d','orHv1000d','clsHv5d','clsHv10d',
                     'clsHv20d','clsHv60d','clsHv90d','clsHv120d','clsHv252d','clsHv500d','clsHv1000d','iv20d','iv30d','iv60d','iv90d','iv6m','clsPx1w',
                     'stkPxChng1wk','clsPx1m','stkPxChng1m','clsPx6m','stkPxChng6m','clsPx1y','stkPxChng1y','divFreq','divYield','divGrwth','divAmt',
                     'correlSpy1m','correlSpy1y','correlEtf1m','correlEtf1y','beta1m','beta1y','ivPctile1m','ivPctile1y','ivPctileSpy','ivPctileEtf','ivStdvMean',
                     'ivStdv1y','ivSpyRatio','ivSpyRatioAvg1m','ivSpyRatioAvg1y','ivSpyRatioStdv1y','ivEtfRatio','ivEtfRatioAvg1m','ivEtfRatioAvg1y','ivEtFratioStdv1y',
                     'ivHvXernRatio','ivHvXernRatio1m','ivHvXernRatio1y','ivHvXernRatioStdv1y','etfIvHvXernRatio','etfIvHvXernRatio1m','etfIvHvXernRatio1y','etfIvHvXernRatioStdv1y',
                     'slopepctile','slopeavg1m','slopeavg1y','slopeStdv1y','etfSlopeRatio','etfSlopeRatioAvg1m','etfSlopeRatioAvg1y','etfSlopeRatioAvgStdv1y','impliedR2',
                     'contango','nextDiv','impliedNextDiv','annActDiv','annIdiv','borrow30','borrow2yr','orHvXern5d',
                     'orHvXern10d','orHvXern20d','orHvXern60d','orHvXern90d','orHvXern120d','orHvXern252d','orHvXern500d','orHvXern1000d',
                     'clsHvXern5d','clsHvXern10d','clsHvXern20d','clsHvXern60d','clsHvXern90d','clsHvXern120d','clsHvXern252d','clsHvXern500d','clsHvXern1000d','iv10d',
                     'iv1yr','fcstSlope','fcstErnEffct','ernMvStdv','impliedEe','impErnMv','impMth2ErnMv','fairVol90d','fairXieeVol90d','fairMth2XieeVol90d','impErnMv90d','impErnMvMth290d',
                     'exErnIv10d','exErnIv20d','exErnIv30d','exErnIv60d','exErnIv90d','exErnIv6m','exErnIv1yr','dlt5Iv10d','dlt5Iv20d','dlt5Iv30d','dlt5Iv60d','dlt5Iv90d',
                     'dlt5Iv6m','dlt5Iv1y','exErnDlt5Iv10d','exErnDlt5Iv20d','exErnDlt5Iv30d','exErnDlt5Iv60d','exErnDlt5Iv90d','exErnDlt5Iv6m','exErnDlt5Iv1y','dlt25Iv10d','dlt25Iv20d',
                     'dlt25Iv30d','dlt25Iv60d','dlt25Iv90d','dlt25Iv6m','dlt25Iv1y','exErnDlt25Iv10d','exErnDlt25Iv20d','exErnDlt25Iv30d','exErnDlt25Iv60d','exErnDlt25Iv90d','exErnDlt25Iv6m',
                     'exErnDlt25Iv1y','dlt75Iv10d','dlt75Iv20d','dlt75Iv30d','dlt75Iv60d','dlt75Iv90d','dlt75Iv6m','dlt75Iv1y','exErnDlt75Iv10d','exErnDlt75Iv20d','exErnDlt75Iv30d',
                     'exErnDlt75Iv60d','exErnDlt75Iv90d','exErnDlt75Iv6m','exErnDlt75Iv1y','dlt95Iv10d','dlt95Iv20d','dlt95Iv30d','dlt95Iv60d','dlt95Iv90d','dlt95Iv6m','dlt95Iv1y',
                     'exErnDlt95Iv10d','exErnDlt95Iv20d','exErnDlt95Iv30d','exErnDlt95Iv60d','exErnDlt95Iv90d','exErnDlt95Iv6m','exErnDlt95Iv1y','fwd30_20','fwd60_30','fwd90_60',
                     'fwd180_90','fwd90_30','fexErn30_20','fexErn60_30','fexErn90_60','fexErn180_90','fexErn90_30','ffwd30_20','ffwd60_30','ffwd90_60','ffwd180_90','ffwd90_30',
                     'ffexErn30_20','ffexErn60_30','ffexErn90_60','ffexErn180_90','ffexErn90_30','fbfwd30_20','fbfwd60_30','fbfwd90_60','fbfwd180_90','fbfwd90_30','fbfexErn30_20',
                     'fbfexErn60_30','fbfexErn90_60','fbfexErn180_90','fbfexErn90_30','impliedEarningsMove']]
        return base
    
    def pca(self, base, numero_componentes, drop_columns):
        X = base.drop(drop_columns, axis=1, errors='ignore')
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA(n_components=numero_componentes)
        X_pca = pca.fit_transform(X_scaled)

        # Print explained variance ratio to help choose n_components
        print("Explained Variance Ratio:", pca.explained_variance_ratio_)
        print("Total Explained Variance:", sum(pca.explained_variance_ratio_))

        X_pca_df = pd.DataFrame(data = X_pca, columns = [f'PC{i+1}' for i in range(numero_componentes)])
        

        # 4. Apply K-Means Clustering
        n_clusters = 10  # Choose the number of clusters. You might need to experiment with this.
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10) # Set random_state for reproducibility
        X_pca_df['Cluster'] = kmeans.fit_predict(X_pca) # Fit and predict in one line. It is better to use the DataFrame here.
        #print(X_pca_df.head())
        centers = kmeans.cluster_centers_
        print("Cluster Centers (in PCA space):\n", centers)

        base['Cluster'] = X_pca_df['Cluster']
        for cluster in base['Cluster'].unique():
            print(f"Cluster {cluster}:\n", base[base['Cluster'] == cluster].describe())
        base.to_pickle("C:\\Users\\gasto\\OneDrive\\base_test_clusters.pkl")
        return None
    
    def calcular_retornos(self, base_cluster, clusters = [2,9]):
        base_total = pd.DataFrame()
        for clus in clusters:
            base_test = base_cluster[base_cluster['Cluster'] == clus]
            retornos = []
            for i,j in zip(base_test['ticker'],base_test['tradeDate']):
                #print(i,j)
                base_ret = base_cluster[(pd.to_datetime(base_cluster['tradeDate']) >= pd.to_datetime(j)) & (base_cluster['ticker'] == i)].sort_values(['tradeDate']).head(5).drop_duplicates()['pxAtmIv'].pct_change(4).iloc[-1]
                retornos.append(base_ret)
            print(retornos)
            retornos = [x for x in retornos if not math.isnan(x)]
            base_estadisticas = pd.DataFrame({
                                            'cluster':[clus],
                                            'count':[len(retornos)],
                                            'min':[np.min(retornos)],
                                            'q5':[np.quantile(retornos, 0.05)],
                                            'q25':[np.quantile(retornos, 0.25)],
                                            'q50':[np.quantile(retornos, 0.50)],
                                            'mean':[np.mean(retornos)],
                                            'q75':[np.quantile(retornos, 0.75)],
                                            'q95':[np.quantile(retornos, 0.95)],
                                            'max':[np.max(retornos)],
                                            'std':[np.std(retornos)],
                                            })
            base_total = pd.concat([base_total, base_estadisticas])
        print(base_total)
        return base_total
    
op = opciones()
## get the data
# base = op.get_core_data_historical(list_dates=['2024-02-20'])
#base = op.read_train_data("C:\\Users\\gasto\\OneDrive\\base_datos_proyecto.pkl")
#op.pca(base = base,numero_componentes=35,drop_columns=['ticker','tradeDate','pxAtmIv'])
base_cluster = pd.read_pickle("C:\\Users\\gasto\\OneDrive\\base_test_clusters.pkl")
print(base_cluster.groupby(['Cluster']).count())
op.calcular_retornos(base_cluster, clusters = [2,5,8,9])