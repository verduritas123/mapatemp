import pandas as pd
import matplotlib.pyplot as plt

# Ruta al archivo CSV (actualiza esta ruta con la tuya)
archivo_csv = r"mediterraneo.csv"

# Cargar el archivo CSV
df = pd.read_csv(archivo_csv)

# Confirmar las primeras filas del DataFrame para verificar los datos
print(df.head())

# Filtrar por la región de interés (latitud 20 a 50, longitud -20 a 40)
lat_min, lat_max = 20, 50  # Latitudes
lon_min, lon_max = -20, 40  # Longitudes
df_region = df[(df['latitude'] >= lat_min) & (df['latitude'] <= lat_max) &
               (df['longitude'] >= lon_min) & (df['longitude'] <= lon_max)]

# Calcular el promedio por década para esta región
promedio_decada_region = df_region.groupby(['decade'])['sst'].mean().reset_index()

# Graficar la temperatura promedio de la región por década
plt.figure(figsize=(10, 6))
plt.plot(promedio_decada_region['decade'], promedio_decada_region['sst'], marker='o', color='blue', label='Región')
plt.title('Temperatura promedio del mar Mediterráneo por década')
plt.xlabel('Década')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid()
plt.show()

# Función para graficar el mapa de temperatura de una década específica
def graficar_mapa(df, decada):
    # Filtrar los datos de la década
    temp_decada = df[df['decade'] == decada]
    
    # Reorganizar los datos para crear una matriz con latitudes y longitudes
    temp_mapa = temp_decada.pivot(index='latitude', columns='longitude', values='sst')
    
    # Crear el mapa
    plt.figure(figsize=(10, 6))
    plt.contourf(temp_mapa.columns, temp_mapa.index, temp_mapa.values, cmap='coolwarm', levels=20)
    plt.colorbar(label='Temperatura (°C)')
    plt.title(f'Mapa de Temperatura Superficial ({decada}s) - Mar Mediterráneo')
    plt.xlabel('Longitud')
    plt.ylabel('Latitud')
    plt.show()

# Graficar un ejemplo de mapa para la década de 1980
graficar_mapa(df_region, 2020)
