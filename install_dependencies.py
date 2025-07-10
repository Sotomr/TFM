#!/usr/bin/env python3
"""
Script de instalación de dependencias para TFM Portfolio Optimization
Instala todas las librerías necesarias y verifica que funcionan correctamente.

Ejecutar como: python install_dependencies.py
"""

import subprocess
import sys
import importlib
import os

def run_command(command, description):
    """Ejecutar comando y manejar errores"""
    print(f"\n🔧 {description}")
    print(f"💻 Ejecutando: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✅ {description} - COMPLETADO")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - ERROR")
        print(f"Error: {e.stderr}")
        return False

def check_installation():
    """Verificar que las librerías principales estén instaladas"""
    print("\n🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 50)
    
    required_packages = {
        'numpy': '1.26.4',
        'pandas': '2.3.0',
        'tensorflow': '2.16.2',
        'pymoo': '0.6.1.5',
        'sklearn': None,
        'yfinance': None,
        'lightgbm': None,
        'matplotlib': None,
        'seaborn': None,
        'joblib': None,
        'scipy': None
    }
    
    all_ok = True
    
    for package, expected_version in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'unknown')
            
            if expected_version and version != expected_version:
                print(f"⚠️  {package}: {version} (esperado: {expected_version})")
            else:
                print(f"✅ {package}: {version}")
                
        except ImportError:
            print(f"❌ {package}: NO INSTALADO")
            all_ok = False
    
    return all_ok

def install_dependencies():
    """Instalar todas las dependencias paso a paso"""
    print("🚀 INSTALACIÓN DE DEPENDENCIAS TFM")
    print("=" * 50)
    
    commands = [
        # Actualizar pip primero
        ("python -m pip install --upgrade pip", "Actualizando pip"),
        
        # Core dependencies
        ("pip install numpy==1.26.4", "Instalando NumPy"),
        ("pip install pandas==2.3.0", "Instalando Pandas"),
        ("pip install scipy>=1.10.0", "Instalando SciPy"),
        ("pip install scikit-learn>=1.3.0", "Instalando Scikit-learn"),
        ("pip install joblib>=1.3.0", "Instalando Joblib"),
        
        # TensorFlow
        ("pip install tensorflow==2.16.2", "Instalando TensorFlow"),
        
        # Financial data
        ("pip install yfinance>=0.2.18", "Instalando yfinance"),
        
        # Optimization
        ("pip install pymoo==0.6.1.5", "Instalando pymoo"),
        
        # Tree models
        ("pip install lightgbm>=4.0.0", "Instalando LightGBM"),
        
        # Plotting
        ("pip install matplotlib>=3.7.0", "Instalando Matplotlib"),
        ("pip install seaborn>=0.12.0", "Instalando Seaborn"),
        
        # PyTorch (CPU version)
        ("pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu", 
         "Instalando PyTorch CPU"),
        ("pip install pytorch-lightning==2.5.2", "Instalando PyTorch Lightning"),
        ("pip install pytorch-forecasting==1.4.0", "Instalando PyTorch Forecasting"),
        
        # Jupyter
        ("pip install jupyter>=1.0.0 notebook>=6.5.0 ipykernel>=6.20.0", 
         "Instalando Jupyter"),
        
        # Data utils
        ("pip install openpyxl>=3.1.0 xlrd>=2.0.0 pyarrow>=12.0.0", 
         "Instalando utilidades de datos")
    ]
    
    failed_commands = []
    
    for command, description in commands:
        if not run_command(command, description):
            failed_commands.append((command, description))
    
    # Resumen
    print("\n📋 RESUMEN DE INSTALACIÓN")
    print("=" * 50)
    
    if failed_commands:
        print("❌ ALGUNOS PAQUETES FALLARON:")
        for cmd, desc in failed_commands:
            print(f"   - {desc}")
        print("\n💡 Intenta instalar manualmente los paquetes que fallaron")
        return False
    else:
        print("✅ TODAS LAS DEPENDENCIAS INSTALADAS CORRECTAMENTE")
        return True

def create_directories():
    """Crear estructura de directorios necesaria"""
    print("\n📁 CREANDO ESTRUCTURA DE DIRECTORIOS")
    
    directories = [
        "data/raw", "data/interim", "data/processed",
        "models", "results", "figures"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ {directory}")

def main():
    """Función principal"""
    print("🎯 TFM PORTFOLIO OPTIMIZATION - SETUP")
    print("=" * 60)
    
    # 1. Crear directorios
    create_directories()
    
    # 2. Instalar dependencias
    if install_dependencies():
        # 3. Verificar instalación
        if check_installation():
            print("\n🎉 ¡INSTALACIÓN COMPLETADA CON ÉXITO!")
            print("\n📝 Próximos pasos:")
            print("   1. Ejecutar: jupyter notebook")
            print("   2. Abrir: 00_test_env.ipynb")
            print("   3. Verificar que todo funciona")
            print("   4. Ejecutar el script de validación: python notebooks/validate_models.py")
        else:
            print("\n⚠️  INSTALACIÓN COMPLETADA CON ADVERTENCIAS")
            print("Revisa los paquetes marcados arriba")
    else:
        print("\n❌ INSTALACIÓN FALLÓ")
        print("Revisa los errores e intenta instalar manualmente")

if __name__ == "__main__":
    main() 