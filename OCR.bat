@echo off
title OCR

:menu
cls
echo Optical Character Recognition For Digits
echo.
echo ====  1. KNN    ====
echo ====  2. Model  ====
echo ====  3. Info   ====
echo ====  4. Exit   ====
echo.
set /p select=">>"
if %select%==1 goto KNN
if %select%==2 goto Model
if %select%==3 goto info
if %select%==4 goto exit

:KNN
cls
start knn_mnist.py
goto menu
pause

:Model
cls
start weights_model.py
goto menu
pause

:info
cls
echo Systemy Sztucznej Inteligencji - Projekt Zaliczeniowy
echo Optical Character Recognition For Digits
echo.
echo -----------------------------------------------------
echo.
echo Wojciech Babinski
echo Informatyka sem IV
pause
goto menu

:exit
echo on
