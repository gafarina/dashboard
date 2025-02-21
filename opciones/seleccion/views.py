from django.shortcuts import render


def seleccion(request):
    #stock_picker = tickers_nifty50()
    #print(stock_picker)
    # va a llamar el html en el template
    return render(request, 'seleccion/base.html')
