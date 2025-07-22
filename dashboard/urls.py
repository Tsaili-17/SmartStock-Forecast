"""
URL configuration for dashboard project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from dashboardapp import views as dashboard_views # Upload, Datatable, Comparison
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    # dashboardapp=>上傳資料到選擇完切分方式
    path('', dashboard_views.home, name="home"),   
    path('upload', dashboard_views.page1_upload, name='upload'),
    path('visualization', dashboard_views.page1_1_visualization, name='visualization'),
    path('datatable', dashboard_views.page2_datatable, name='datatable'),
    path('train_test_comparison', dashboard_views.page3_train_test_comparison, name='train_test_comparison'),
    path('train_test_comparison', dashboard_views.page3_train_test_comparison, name='page3_train_test_comparison'),
    path('bert_model', dashboard_views.page4_bert_model, name='page4_bert_model'),

]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

