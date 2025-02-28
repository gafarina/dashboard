from django.shortcuts import render
import yfinance as yf


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
    return render(request, 'seleccion/base.html',{'current_prices':current_prices})
