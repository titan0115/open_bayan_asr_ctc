@echo off
echo Запуск TensorBoard...
call venv\Scripts\activate.bat
tensorboard --logdir=runs --port=6006
pause
