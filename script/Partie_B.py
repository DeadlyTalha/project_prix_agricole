import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# choix du nbre de clusters
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])

sil_scores = {}
for k in range(2, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    sil_scores[k] = silhouette_score(X_scaled, km.fit_predict(X_scaled))
best_k = max(sil_scores, key=sil_scores.get)  
print(f"Meilleur k : {best_k} ")
    # ajout de la var cluster 
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['cluster'] = km_final.fit_predict(X_scaled)

# modele avec cluster
feature_cols_clust = feature_cols + ['cluster']
X_clust = df[feature_cols_clust]
y       = df['target_stage1_tabular_score']

X_tr, X_te, y_tr, y_te = train_test_split(
    X_clust, y, test_size=0.2, random_state=42, shuffle=False
)
rf_clust = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_clust.fit(X_tr, y_tr)
y_pred = rf_clust.predict(X_te)

print(f"MAE  : {mean_absolute_error(y_te, y_pred):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_te, y_pred)):.4f}")
print(f"R²   : {r2_score(y_te, y_pred):.4f}")