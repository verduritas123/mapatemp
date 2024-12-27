from eccodes import codes_grib_new_from_file, codes_get, codes_get_array, codes_release
import pandas as pd

# Ruta al archivo GRIB (actualiza esta ruta con la tuya)
archivo_grib = r"C:\Users\eleni\Downloads\361b6f66e935b0570c67edfd82c99c61.grib"

# Filtrar por región (activa/desactiva esta sección según lo necesites)
# Define los límites de latitud y longitud para filtrar (ejemplo: Océano Atlántico)
usar_filtro = True  # Cambia a False si no quieres filtrar por región
lat_min, lat_max = -50, 50    # Latitud mínima y máxima
lon_min, lon_max = -80, 20    # Longitud mínima y máxima

# Procesar el archivo GRIB
try:
    with open(archivo_grib, 'rb') as f:
        contador = 0  # Para identificar bloques
        print("Procesando archivo GRIB...")

        while True:
            # Leer el siguiente mensaje GRIB
            mensaje = codes_grib_new_from_file(f)
            if mensaje is None:
                break  # Fin del archivo
            
            # Extraer las variables clave
            latitudes = codes_get_array(mensaje, 'latitudes')
            longitudes = codes_get_array(mensaje, 'longitudes')
            valores = codes_get_array(mensaje, 'values')
            tiempo = codes_get(mensaje, 'dataDate')  # Fecha asociada al mensaje
            
            # Si usamos filtro, aplicar las condiciones de latitud y longitud
            if usar_filtro:
                filtro = (latitudes >= lat_min) & (latitudes <= lat_max) & \
                         (longitudes >= lon_min) & (longitudes <= lon_max)
                latitudes = latitudes[filtro]
                longitudes = longitudes[filtro]
                valores = valores[filtro]
            
            # Crear un DataFrame con los datos procesados
            df = pd.DataFrame({
                'latitud': latitudes,
                'longitud': longitudes,
                'temperatura': valores,
                'fecha': [tiempo] * len(valores)
            })
            
            # Guardar cada bloque como un archivo CSV
            df.to_csv(f"datos_temperatura_bloque_{contador}.csv", index=False)
            print(f"Bloque {contador} procesado y guardado: datos_temperatura_bloque_{contador}.csv")
            contador += 1
            
            # Liberar el mensaje para liberar memoria
            codes_release(mensaje)

        print("Procesamiento completo. Los datos se han guardado en bloques CSV.")

except Exception as e:
    print("Se produjo un error durante el procesamiento:", e)