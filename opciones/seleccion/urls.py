from django.urls import path, include
# se imporan las views
from .import views
urlpatterns = [
    #
    #path('', views.stockPicker, name = 'stockpicker'),
    #Este es el path de la applicacion que es la base
    path('', views.positions, name = 'seleccion'),
]