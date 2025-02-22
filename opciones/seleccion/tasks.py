from celery import shared_task
import yfinance as yf
@shared_task(bind = True)
def update_stock(self):
    stock_symbols = ["AAPL", "MSFT", "GOOG", "AMZN"]
    data = yf.download(stock_symbols, period="1d")
    current_prices = data['Close'].transpose().reset_index()
    date = current_prices.columns[-1]
    current_prices['date'] = date
    print("-----------")
    print(current_prices)
    current_prices.columns = ['ticker','price','date']
    current_prices = current_prices[['date','ticker','price']]
    data = current_prices.to_dict('records')
    return 'Done'