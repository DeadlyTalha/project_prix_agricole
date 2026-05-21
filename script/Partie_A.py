import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv('data/ex24_prix_agricoles.csv')
df.head()

drop_cols = ['record_id','governorate','city','region','date','data_nature','target_stage1_tabular_score','target_stage2_lstm_score','lstm_agri_price_next_3m']
feature_cols = [c for c in df.columns if c not in drop_cols and df[c].dtype != object]
target_col = 'target_stage1_tabular_score'

X = df[feature_cols]
y = df[target_col]

# split 
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# entrenement 
model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# metriques
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# importance des features
fi = pd.DataFrame({'feature': feature_cols, 'importance': model.feature_importances_})
print(fi.sort_values('importance', ascending=False).head(10).to_string(index=False))

