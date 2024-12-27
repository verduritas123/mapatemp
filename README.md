# Visualización y Modelado de Temperaturas

## Descripción

Este proyecto modela las temperaturas del Mar Mediterráneo utilizando datos satelitales. Incluye procesamiento, visualización de datos y predicciones futuras con algoritmos como la Regresión Polinómica y ARIMA.

## Ejecución

### Procesar datos:
Convierte archivos GRIB a formato CSV ejecutando:

python 'gribacsv.py'


### Modelar temperaturas:
Genera mapas visuales y analiza tendencias ejecutando:

python 'prueba.py'

### Generar predicciones:
Realiza predicciones de temperatura basadas en las sugerencias del profesor:

python 'prediccion.py'

## Archivos del Proyecto

'gribacsv.py': Convierte archivos GRIB a CSV, filtrando por región.

'prueba.py': Modela las temperaturas y genera mapas.

'prediccion.py': Genera predicciones finales de temperatura y visualizaciones.

'reg.py', 'reg2.py', 'reg3.py': Implementan Regresión Polinómica.

'arima.py': Aplica el modelo ARIMA para series temporales.

## Requisitos

    Python 3.x

    Bibliotecas necesarias:

        numpy

        pandas

        matplotlib

        scikit-learn

        statsmodels

        eccodes