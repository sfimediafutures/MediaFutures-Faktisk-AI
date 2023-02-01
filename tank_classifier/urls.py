from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.index, name='index'),
    path('tank/', views.tank_classifier_view, name='tankclassifier-home'),
    path('lang/', views.langauge_classifier_view, name='languageclassification-home'),
    path('tutorial/', views.tutorial, name='tutorial-home'),
    path('test', views.test, name='test'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)