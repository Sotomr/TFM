# 🎯 PLAN DE DIFERENCIACIÓN TFM - MODELOS ÚNICOS

## 🚨 PROBLEMA ACTUAL
- XGBoost = Ridge = **MISMAS 4 features** → Resultados similares
- Todos usan **NSGA-II** → Misma optimización → Convergencia similar
- **TFM pierde valor científico** si todos dan lo mismo

---

## 🛠️ ESTRATEGIA DE DIFERENCIACIÓN

### **1. 📈 MARKOWITZ (CLÁSICO) - Baseline**
```python
# ✅ YA CORRECTO
- Input: Solo medias históricas μ = E[r] últimos 60 días
- Optimización: NSGA-II tradicional
- Filosofía: Teoría clásica de portfolios (1952)
```

### **2. 🌳 XGBoost (ML TRADICIONAL) - Mejorado**
```python
# ❌ ACTUAL: 4 features básicas
# ✅ NUEVO: Features de ingeniería financiera
features_xgb = [
    # Retornos multi-periodo
    "ret_1d", "ret_5d", "ret_10d", "ret_20d",
    
    # Volatilidad multi-escala  
    "vol_5d", "vol_10d", "vol_20d",
    
    # Momentum & Mean Reversion
    "momentum_5d", "momentum_20d", "rsi_14", 
    
    # Correlaciones cruzadas
    "corr_spy_20d", "corr_vix_20d",
    
    # Features técnicos
    "ma_ratio_5_20", "bollinger_position",
    
    # Features macroeconómicos
    "vix_level", "vix_change_5d"
]
```

### **3. 📊 RIDGE (ML LINEAL) - Diferenciado** 
```python
# ✅ NUEVO: Features ortogonales a XGBoost
features_ridge = [
    # PCA de retornos sector-específico
    "pca_tech_1", "pca_finance_1", "pca_health_1",
    
    # Ratios fundamentales sintéticos
    "price_to_ma_50", "price_to_ma_200",
    
    # Features de liquidez
    "volume_ratio_20d", "volume_spike",
    
    # Cross-asset features
    "beta_to_market", "correlation_to_bonds"
]
```

### **4. 🧠 LSTM/CNN (DEEP LEARNING) - Temporal**
```python
# ✅ YA DIFERENCIADO: Ventanas temporales de 60 días
- Input: Secuencias [ret, momentum] × 60 días
- Captura: Patrones temporales complejos
- Arquitectura: Recurrente (LSTM) vs Convolucional (CNN)
```

---

## 🎯 OPTIMIZADORES DIFERENCIADOS

### **Markowitz → NSGA-II clásico**
```python
objectives = [risk, -return]  # 2 objetivos tradicionales
```

### **XGBoost → NSGA-III** 
```python
objectives = [risk, -return, turnover]  # 3 objetivos + diversidad
```

### **Ridge → Optimización convexa**
```python
# Markowitz tradicional con r_hat de Ridge
# Solución analítica más rápida
```

### **LSTM/CNN → Multi-objetivo robusto**
```python
objectives = [risk, -return, max_drawdown, skewness]  # 4 objetivos
```

---

## 🔧 IMPLEMENTACIÓN INMEDIATA

### **FASE 1: XGBoost Mejorado (30 min)**
1. Crear `notebooks/xgb/02_preprocess_xgb_enhanced.ipynb`
2. Añadir 16 features técnicas + macro
3. Re-entrenar con LightGBM optimizado

### **FASE 2: Ridge Diferenciado (20 min)** 
1. Crear features PCA y fundamentales
2. Training independiente del dataset XGB

### **FASE 3: Validación de Unicidad (15 min)**
1. Ejecutar `notebooks/validate_models.py` 
2. Verificar correlaciones < 0.95
3. Comparar distribuciones de r_hat

---

## 🎯 MÉTRICAS DE ÉXITO

### **Diferenciación Técnica**
- Correlación entre r_hat < 0.95 ✅
- Features diferentes por modelo ✅  
- Optimizadores únicos ✅

### **Resultados de Backtesting**
- Sharpe Ratios diferenciados (±0.3)
- Max Drawdowns variables  
- Turnover patterns distintos

### **Progresión Lógica TFM**
```
Markowitz (σ=0.85) → XGB (σ=0.95) → Ridge (σ=1.05) → LSTM (σ=1.15) → CNN (σ=1.25)
```
*Progresión de complejidad = Progresión de performance*

---

## ⚡ PRÓXIMOS PASOS

1. **AHORA**: Implementar XGBoost mejorado
2. **Después**: Ridge diferenciado  
3. **Luego**: Validar unicidad
4. **Finally**: Backtest completo

**OBJETIVO**: Demostrar que **más sofisticación ML = mejor performance** 