from django.urls import path
from django.utils.regex_helper import normalize
from . import views

app_name = 'shop'

urlpatterns = [
    path('', views.home, name='home'),
    path('shop/', views.shop, name='shop'),
    path('shop/<slug:category_slug>/', views.shop, name='categries'),
    path('shop/<slug:category_slug>/<slug:product_details_slug>/', views.product_details, name='product_details'),
    path('run_python_code/', views.run_python_code, name='run_python_code'),
    # path('run_python_code/<int:product_id>/', views.run_python_code, name='run_python_code'),
    path('search/', views.search, name='search'),
    path('review/<int:product_id>/', views.review, name='review'),
]