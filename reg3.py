import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Cargar el archivo CSV con los datos de temperatura
df = pd.read_csv("mediterraneo.csv")

# Verificar las columnas disponibles en el DataFrame
print("Columnas disponibles:", df.columns)

# Filtrar los datos por la región de interés
lat_min, lat_max = 20, 50
lon_min, lon_max = -20, 40
df = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
        (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]

# Agrupar por década y obtener el promedio de la temperatura
df_decada = df.groupby('decade')['sst'].mean().reset_index()  # Promedio de la temperatura por década

# Verificar los primeros registros del DataFrame para asegurarse de que todo esté bien
print(df_decada.head())

# Definir las variables independientes (X) y dependientes (y)
X = df_decada['decade'].values.reshape(-1, 1)  # Décadas como variable independiente
y = df_decada['sst'].values  # Temperaturas como variable dependiente

# Crear una transformación polinómica (grado 2 en este caso)
poly = PolynomialFeatures(degree=2)  # Ajusta el grado según sea necesario
X_poly = poly.fit_transform(X)

# Ajustar el modelo de regresión polinómica
model = LinearRegression()
model.fit(X_poly, y)

# Hacer predicciones para las décadas futuras (hasta 2100)
future_decades = np.array(range(df_decada['decade'].iloc[-1] + 10, 2120, 10)).reshape(-1, 1)
future_decades_poly = poly.transform(future_decades)
forecast = model.predict(future_decades_poly)

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.scatter(df_decada['decade'], df_decada['sst'], color='blue', label='Datos Observados', marker='o')
plt.plot(df_decada['decade'], model.predict(X_poly), color='green', label='Ajuste Polinómico', linestyle='-')
plt.plot(future_decades, forecast, label='Predicción Polinómica', linestyle='--', color='red', marker='x')
plt.title('Predicción Polinómica de la Temperatura Promedio del Mar Mediterráneo')
plt.xlabel('Década')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid()
plt.show()

# Mostrar las predicciones
for decada, temp in zip(future_decades.flatten(), forecast):
    print(f"Predicción para {decada}: {temp:.2f} °C")
