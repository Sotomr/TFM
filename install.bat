@echo off
echo =========================================
echo TFM Portfolio Optimization - Setup
echo =========================================
echo.

echo 🔧 Actualizando pip...
python -m pip install --upgrade pip

echo.
echo 🔧 Instalando dependencias principales...
pip install numpy==1.26.4 pandas==2.3.0 scipy>=1.10.0 scikit-learn>=1.3.0 joblib>=1.3.0

echo.
echo 🔧 Instalando TensorFlow...
pip install tensorflow==2.16.2

echo.
echo 🔧 Instalando herramientas financieras...
pip install yfinance>=0.2.18

echo.
echo 🔧 Instalando optimización...
pip install pymoo==0.6.1.5

echo.
echo 🔧 Instalando modelos ML...
pip install lightgbm>=4.0.0

echo.
echo 🔧 Instalando visualización...
pip install matplotlib>=3.7.0 seaborn>=0.12.0

echo.
echo 🔧 Instalando PyTorch (CPU)...
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
pip install pytorch-lightning==2.5.2 pytorch-forecasting==1.4.0

echo.
echo 🔧 Instalando Jupyter...
pip install jupyter>=1.0.0 notebook>=6.5.0 ipykernel>=6.20.0

echo.
echo 🔧 Instalando utilidades...
pip install openpyxl>=3.1.0 xlrd>=2.0.0 pyarrow>=12.0.0

echo.
echo ✅ Instalación completada!
echo.
echo 📝 Próximos pasos:
echo    1. Ejecutar: jupyter notebook
echo    2. Abrir: 00_test_env.ipynb  
echo    3. Verificar que todo funciona
echo.
pause 