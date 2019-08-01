"""main URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
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
from . import views

urlpatterns = [
    path('', views.home, name="start_game"),
    path('game/', views.play_game, name="play_game"),
    path('game/round/<int:game_round_no>/', views.play_round, name="play_round"),
    path('game/round/<int:game_round_no>/prediction/', views.get_prediction, name="get_prediction"),
    path('game/round/<int:game_round_no>/prediction/<int:prediction>/', views.receive_prediction, name="receive_prediction"),
    path('game/round/<int:game_round_no>/trick/', views.get_play, name="get_play"),
    path('game/round/<int:game_round_no>/trick/<int:trick_card>/', views.receive_play, name="receive_play"),
    path('game/round/<int:game_round_no>/result/', views.show_result, name="show_result"),
    path('game/end/', views.end, name="end")
]
