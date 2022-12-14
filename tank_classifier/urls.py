from django.urls import path
from . import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    path('', views.tank_classifier_view, name='tankclassifier-home'),
    path('check_langauge/', views.langauge_classifier_view, name='languageclassification-home'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)