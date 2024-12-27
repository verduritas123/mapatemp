import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Cargar el archivo CSV con los datos de temperatura
df = pd.read_csv("mediterraneo.csv")

# Filtrar los datos por la región de interés
lat_min, lat_max = 20, 50
lon_min, lon_max = -20, 40
df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
        (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]

# Convertir la columna 'decade' a tipo numérico para hacer la regresión
df['decade'] = (df['decade'] // 10) * 10

# Calcular la temperatura promedio por década
df_decada = df.groupby('decade')['sst'].mean().reset_index()

# Verifica que los datos de entrada son variados
print(df_decada.head())

# Definir las variables independientes (X) y dependientes (y)
X = df_decada['decade'].values.reshape(-1, 1)  # Décadas como variable independiente
y = df_decada['sst'].values  # Temperaturas como variable dependiente

# Transformar las características (decada) en características polinómicas
poly = PolynomialFeatures(degree=3)  # Grado 3, puedes ajustar esto
X_poly = poly.fit_transform(X)

# Crear el modelo de regresión lineal
model = LinearRegression()
model.fit(X_poly, y)

# Hacer predicciones sobre los datos de la misma década (ajuste del modelo)
y_pred = model.predict(X_poly)

# Evaluar el modelo
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Error cuadrático medio (MSE): {mse:.4f}")
print(f"Coeficiente de determinación (R^2): {r2:.4f}")

# Predecir las temperaturas para las próximas décadas (hasta 2100)
futuras_decadas = np.arange(2020, 2101, 10).reshape(-1, 1)
futuras_decadas_poly = poly.transform(futuras_decadas)
predicciones = model.predict(futuras_decadas_poly)

# Graficar los resultados
plt.figure(figsize=(10, 6))

# Graficar las temperaturas observadas
plt.plot(df_decada['decade'], df_decada['sst'], marker='o', label='Temperaturas Observadas', color='blue')

# Graficar la curva ajustada
plt.plot(df_decada['decade'], y_pred, color='red', label='Ajuste Polinómico', linestyle='--')

# Graficar las predicciones para futuras décadas
plt.plot(futuras_decadas, predicciones, color='green', label='Predicciones Futuras', linestyle='-.', marker='x')

plt.title('Predicción de la Temperatura Promedio del Mar Mediterráneo (Regresión Polinómica)')
plt.xlabel('Década')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.show()

# Mostrar las predicciones para las próximas décadas
for decada, temp in zip(futuras_decadas.flatten(), predicciones):
    print(f'Predicción para la década de {decada}: {temp:.2f} °C')
