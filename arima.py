import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

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

# Ajustar el modelo ARIMA
model = ARIMA(y, order=(2, 1, 0))  # Parámetros p, d, q
model_fit = model.fit()

# Hacer predicciones a futuro (hasta 2100)
forecast = model_fit.forecast(steps=9)  # Predicción para 9 décadas futuras
future_decades = list(range(df_decada['decade'].iloc[-1] + 10, 2120, 10))  # Décadas futuras

# Graficar resultados
plt.figure(figsize=(10, 6))
plt.plot(df_decada['decade'], df_decada['sst'], label='Datos Observados', marker='o', color='blue')
plt.plot(future_decades, forecast, label='Predicción ARIMA', linestyle='--', color='red', marker='x')
plt.title('Predicción de la Temperatura Promedio del Mar Mediterráneo (ARIMA)')
plt.xlabel('Década')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid()
plt.show()

# Mostrar predicciones
for decada, temp in zip(future_decades, forecast):
    print(f"Predicción para {decada}: {temp:.2f} °C")
