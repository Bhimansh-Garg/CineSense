from django.urls import path
from . import views
urlpatterns = [
    path('', views.review_list, name='review_list'),
    path('create/', views.review_create, name='review_create'),
    path('<int:review_id>/edit/', views.review_edit, name='review_edit'),
    path('<int:review_id>/analyze/', views.review_analyse, name='review_analyse'),
    path('<int:review_id>/delete/', views.review_delete, name='review_delete'),
    path('register/', views.register, name='register'),
] 