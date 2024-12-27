import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

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

# Predecir las temperaturas para las próximas décadas (hasta 2100)
futuras_decadas = np.arange(2020, 2101, 10)  # Desde 2020 hasta 2100, con intervalos de 10 años
futuras_decadas_poly = poly.transform(futuras_decadas.reshape(-1, 1))  # Transformar las futuras décadas
predicciones = model.predict(futuras_decadas_poly)  # Realizar las predicciones



# Graficar un diagrama de barras de las predicciones de temperatura para las décadas futuras
plt.figure(figsize=(12, 7))

# Graficar las predicciones para futuras décadas como un diagrama de barras
bars = plt.bar(futuras_decadas, predicciones, color='royalblue', edgecolor='black', width=6, label='Predicción de Temperatura')

# Mejorar el aspecto del gráfico
plt.title('Predicción de la Temperatura Promedio del Mar Mediterráneo (Futuras Décadas)', fontsize=16, fontweight='bold')
plt.xlabel('Década', fontsize=14)
plt.ylabel('Temperatura (°C)', fontsize=14)
plt.xticks(futuras_decadas, rotation=45)
plt.yticks(np.arange(min(predicciones) - 1, max(predicciones) + 1, 1))  # Ajustar la escala y marcar cada 0.5°C

# Agregar etiquetas de texto encima de cada barra
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f'{yval:.2f}', ha='center', va='bottom', fontsize=12, color='black')

# Agregar la leyenda
plt.legend(loc='upper left')

# Añadir una cuadrícula para mayor claridad
plt.grid(True, linestyle='--', alpha=0.7)

# Añadir sombra a las barras
for bar in bars:
    bar.set_zorder(3)
    bar.set_linewidth(1.5)

# Mejorar la presentación y hacer que el gráfico se ajuste bien
plt.tight_layout()

# Mostrar el gráfico
plt.show()

# Mostrar las predicciones para las próximas décadas
for decada, temp in zip(futuras_decadas, predicciones):
    print(f'Predicción para la década de {decada}: {temp:.2f} °C')
