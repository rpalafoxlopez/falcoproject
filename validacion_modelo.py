"""
validacion_modelo.py
Validación formal del modelo mejorado para el Mundial 2026
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.stats import poisson
from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURACIÓN
# ============================================================================

# Fechas de corte (usamos el rango de la validación externa)
TRAIN_END = "2025-09-01"
TEST_START = "2025-09-02"
TEST_END = "2026-06-01"  # Incluye los partidos previos al Mundial

# ============================================================================
# 2. CARGA DE DATOS
# ============================================================================

RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
raw = pd.read_csv(RESULTS_URL, parse_dates=["date"])
print(f"📊 Total de partidos: {len(raw):,}")

# ============================================================================
# 3. FUNCIONES DEL MODELO (adaptadas de app.py)
# ============================================================================

def ajustar_por_pausas_hidratacion(lam_h, lam_a):
    """Ajuste por 4 tiempos con pausas de hidratación"""
    factor_general = 0.92
    media_h_general = 1.35
    media_a_general = 1.05
    factor_contraccion = 0.15
    
    lam_h_aj = lam_h * factor_general
    lam_a_aj = lam_a * factor_general
    
    lam_h_aj = lam_h_aj * (1 - factor_contraccion) + media_h_general * factor_contraccion
    lam_a_aj = lam_a_aj * (1 - factor_contraccion) + media_a_general * factor_contraccion
    
    return lam_h_aj, lam_a_aj

def entrenar_xgboost(hist, FEATURES):
    """Entrena modelo XGBoost con Elo + forma reciente"""
    import xgboost as xgb
    
    K_ELO = 20.0
    ELO_INIT = 1500.0
    
    # Calcular Elo
    ratings = {}
    elo_h, elo_a = [], []
    for _, row in hist.iterrows():
        rh = ratings.get(row.home_team, ELO_INIT)
        ra = ratings.get(row.away_team, ELO_INIT)
        elo_h.append(rh); elo_a.append(ra)
        
        exp_h = 1.0 / (1.0 + 10 ** ((ra - rh) / 400.0))
        if row.home_score > row.away_score: score = 1.0
        elif row.home_score == row.away_score: score = 0.5
        else: score = 0.0
        
        margin = abs(row.home_score - row.away_score)
        delta = K_ELO * (np.log(margin + 1) + 1.0) * (score - exp_h)
        ratings[row.home_team] = rh + delta
        ratings[row.away_team] = ra - delta
    
    hist = hist.copy()
    hist["elo_home"], hist["elo_away"] = elo_h, elo_a
    
    # Calcular forma reciente
    records = {}
    gf10_h, ga10_h, form5_h = [], [], []
    gf10_a, ga10_a, form5_a = [], [], []
    
    for _, row in hist.iterrows():
        h_rec = records.get(row.home_team, [])
        a_rec = records.get(row.away_team, [])
        
        def summarize(hist_rec):
            last10, last5 = hist_rec[-10:], hist_rec[-5:]
            gf = np.mean([x[1] for x in last10]) if last10 else np.nan
            ga = np.mean([x[2] for x in last10]) if last10 else np.nan
            pts = sum(x[3] for x in last5) if last5 else np.nan
            return gf, ga, pts
        
        hgf, hga, hpts = summarize(h_rec)
        agf, aga, apts = summarize(a_rec)
        gf10_h.append(hgf); ga10_h.append(hga); form5_h.append(hpts)
        gf10_a.append(agf); ga10_a.append(aga); form5_a.append(apts)
        
        h_pts = 3 if row.home_score > row.away_score else (1 if row.home_score == row.away_score else 0)
        a_pts = 3 if row.away_score > row.home_score else (1 if row.home_score == row.away_score else 0)
        records.setdefault(row.home_team, []).append((row.date, row.home_score, row.away_score, h_pts))
        records.setdefault(row.away_team, []).append((row.date, row.away_score, row.home_score, a_pts))
    
    hist["gf10_h"], hist["ga10_h"], hist["form5_h"] = gf10_h, ga10_h, form5_h
    hist["gf10_a"], hist["ga10_a"], hist["form5_a"] = gf10_a, ga10_a, form5_a
    
    final_form = records
    final_elo = ratings
    
    # Crear formato largo
    def to_long(df):
        TOURNAMENT_WEIGHTS = {"FIFA World Cup": 4.0, "FIFA World Cup qualification": 2.0, "Friendly": 0.5}
        DEFAULT_WEIGHT = 1.0
        df["tournament_weight"] = df.tournament.map(TOURNAMENT_WEIGHTS).fillna(DEFAULT_WEIGHT)
        
        home_rows = pd.DataFrame({
            "team": df.home_team, "goals": df.home_score, "is_home": 1,
            "elo_team": df.elo_home, "elo_opponent": df.elo_away,
            "gf10": df.gf10_h, "ga10": df.ga10_h, "form5": df.form5_h,
            "tournament_weight": df.tournament_weight,
        })
        away_rows = pd.DataFrame({
            "team": df.away_team, "goals": df.away_score, "is_home": 0,
            "elo_team": df.elo_away, "elo_opponent": df.elo_home,
            "gf10": df.gf10_a, "ga10": df.ga10_a, "form5": df.form5_a,
            "tournament_weight": df.tournament_weight,
        })
        long = pd.concat([home_rows, away_rows], ignore_index=True)
        long["elo_diff"] = long.elo_team - long.elo_opponent
        return long.dropna(subset=["gf10", "ga10", "form5"])
    
    long_df = to_long(hist)
    FEATURES = ["elo_team", "elo_opponent", "elo_diff", "is_home", "gf10", "ga10", "form5", "tournament_weight"]
    
    xgb_model = xgb.XGBRegressor(
        objective="count:poisson", n_estimators=200, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42,
        n_jobs=1
    )
    xgb_model.fit(long_df[FEATURES], long_df["goals"])
    
    return xgb_model, FEATURES, final_elo, final_form

def predecir_partido(xgb_model, FEATURES, final_elo, final_form, home_team, away_team, 
                     use_hydration=True, max_goals=8):
    """Predice un partido con el modelo XGBoost"""
    ELO_INIT = 1500.0
    
    def get_snapshot(team):
        hist_team = final_form.get(team, [])
        last10, last5 = hist_team[-10:], hist_team[-5:]
        gf = np.mean([x[1] for x in last10]) if last10 else 0.0
        ga = np.mean([x[2] for x in last10]) if last10 else 0.0
        pts = sum(x[3] for x in last5) if last5 else 0.0
        elo = final_elo.get(team, ELO_INIT)
        return elo, gf, ga, pts
    
    elo_h, gf_h, ga_h, pts_h = get_snapshot(home_team)
    elo_a, gf_a, ga_a, pts_a = get_snapshot(away_team)
    
    row_home = {"elo_team": elo_h, "elo_opponent": elo_a, "elo_diff": elo_h - elo_a,
                "is_home": 1, "gf10": gf_h, "ga10": ga_h, "form5": pts_h,
                "tournament_weight": 4.0}
    row_away = {"elo_team": elo_a, "elo_opponent": elo_h, "elo_diff": elo_a - elo_h,
                "is_home": 0, "gf10": gf_a, "ga10": ga_a, "form5": pts_a,
                "tournament_weight": 4.0}
    
    feat_df = pd.DataFrame([row_home, row_away])[FEATURES]
    lam_h, lam_a = xgb_model.predict(feat_df)
    
    if use_hydration:
        lam_h, lam_a = ajustar_por_pausas_hidratacion(lam_h, lam_a)
    
    goals = np.arange(0, max_goals + 1)
    score_matrix = np.outer(poisson.pmf(goals, lam_h), poisson.pmf(goals, lam_a))
    score_matrix = score_matrix / score_matrix.sum()
    
    return score_matrix, lam_h, lam_a

def evaluar_predicciones(predictions, test_data):
    """Evalúa las predicciones contra los resultados reales"""
    from sklearn.metrics import log_loss
    import numpy as np
    
    resultados = []
    for pred, real in zip(predictions, test_data):
        # Accuracy 1X2
        pred_1x2 = np.argmax([np.sum(pred[:3, :3]), np.sum(np.diag(pred[:3, :3])), np.sum(pred[:3, 1:])])
        real_1x2 = 0 if real['home_score'] > real['away_score'] else 1 if real['home_score'] == real['away_score'] else 2
        resultados.append(pred_1x2 == real_1x2)
    
    accuracy = np.mean(resultados)
    return accuracy

# ============================================================================
# 4. VALIDACIÓN COMPLETA
# ============================================================================

print("\n" + "="*60)
print("🔬 VALIDACIÓN FORMAL DEL MODELO MEJORADO")
print("="*60)

# Dividir datos
train_data = raw[(raw["date"] <= TRAIN_END)].copy()
test_data = raw[(raw["date"] >= TEST_START) & (raw["date"] <= TEST_END)].copy()

print(f"\n📅 Entrenamiento: {TRAIN_END} ({len(train_data):,} partidos)")
print(f"📅 Prueba: {TEST_START} → {TEST_END} ({len(test_data):,} partidos)")

# Entrenar modelo base
print("\n⚙️ Entrenando modelo base...")
xgb_model, FEATURES, final_elo, final_form = entrenar_xgboost(train_data, [])

# ============================================================================
# 5. EVALUACIÓN POR VERSIÓN
# ============================================================================

print("\n" + "="*60)
print("📊 RESULTADOS DE VALIDACIÓN")
print("="*60)

# Versión A: Base (sin pausas)
print("\n🔵 VERSIÓN A: Modelo base (sin pausas)")
predicciones_a = []
for _, row in test_data.iterrows():
    sm, _, _ = predecir_partido(xgb_model, FEATURES, final_elo, final_form, 
                                row['home_team'], row['away_team'], 
                                use_hydration=False)
    predicciones_a.append(sm)

acc_a = evaluar_predicciones(predicciones_a, test_data.to_dict('records'))
print(f"   Accuracy 1X2: {acc_a*100:.1f}%")

# Versión B: Con pausas de hidratación
print("\n🟢 VERSIÓN B: Modelo base + pausas de hidratación (4 tiempos)")
predicciones_b = []
for _, row in test_data.iterrows():
    sm, _, _ = predecir_partido(xgb_model, FEATURES, final_elo, final_form, 
                                row['home_team'], row['away_team'], 
                                use_hydration=True)
    predicciones_b.append(sm)

acc_b = evaluar_predicciones(predicciones_b, test_data.to_dict('records'))
print(f"   Accuracy 1X2: {acc_b*100:.1f}%")

# ============================================================================
# 6. COMPARATIVA
# ============================================================================

print("\n" + "="*60)
print("📈 COMPARATIVA")
print("="*60)

print(f"""
┌─────────────────────┬──────────────┐
│ Modelo              │ Accuracy 1X2 │
├─────────────────────┼──────────────┤
│ A. Base             │ {acc_a*100:.1f}%          │
│ B. Con pausas       │ {acc_b*100:.1f}%          │
│ Mejora              │ { (acc_b - acc_a)*100:+.1f} pp        │
└─────────────────────┴──────────────┘
""")

# Estimación vs benchmark externo
benchmark = 0.593  # 59.3% del modelo copiado
print(f"\n📌 Benchmark externo (XGBoost base): {benchmark*100:.1f}%")
print(f"📌 Tu modelo con pausas: {acc_b*100:.1f}%")
print(f"📌 Diferencia: {(acc_b - benchmark)*100:+.1f} pp")

# ============================================================================
# 7. GUARDAR RESULTADOS
# ============================================================================

resultados = {
    'fecha_ejecucion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'train_end': TRAIN_END,
    'test_start': TEST_START,
    'test_end': TEST_END,
    'partidos_test': len(test_data),
    'accuracy_base': acc_a,
    'accuracy_con_pausas': acc_b,
    'mejora': acc_b - acc_a,
    'benchmark_externo': benchmark,
    'diferencia_benchmark': acc_b - benchmark,
}

# Mostrar resumen
print("\n" + "="*60)
print("✅ VALIDACIÓN COMPLETADA")
print("="*60)
print(f"\n📊 Resumen:")
print(f"   • Partidos de prueba: {len(test_data)}")
print(f"   • Accuracy base: {acc_a*100:.1f}%")
print(f"   • Accuracy con pausas: {acc_b*100:.1f}%")
print(f"   • Mejora: {(acc_b - acc_a)*100:+.1f} pp")
print(f"   • vs benchmark externo: {(acc_b - benchmark)*100:+.1f} pp")

# Guardar CSV
df_resultados = pd.DataFrame([resultados])
df_resultados.to_csv('resultados_validacion.csv', index=False)
print("\n📁 Resultados guardados en 'resultados_validacion.csv'")
