@echo off
title OCR

:menu
cls
echo Optical Character Recognition For Digits
echo.
echo ====  1. KNN    ====
echo ====  2. Model  ====
echo ====  3. DNN    ====
echo ====  4. Info   ====
echo ====  5. Exit   ====
echo.
set /p select=">>"
if %select%==1 goto KNN
if %select%==2 goto Model
if %select%==3 goto DNN
if %select%==4 goto info
if %select%==5 goto exit

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

:DNN
cls
start neural_network.py
goto menu
pause

:info
cls
echo Zastosowania Sztucznej Inteligencji - Projekt Zaliczeniowy
echo Optical Character Recognition For Digits
echo.
echo -----------------------------------------------------
echo.
echo Wojciech Babinski
echo Informatyka sem VII
pause
goto menu

:exit
echo on
