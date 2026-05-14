# Predictive Maintenance Project

Proyecto de Machine Learning orientado a mantenimiento predictivo en maquinaria industrial.

El objetivo es predecir si una maquina puede fallar a partir de variables operativas como temperatura, velocidad de rotacion, torque y desgaste de herramienta.

## Contexto

En una empresa manufacturera, una averia inesperada puede provocar paradas de produccion, retrasos y costes de reparacion.

Este proyecto plantea una primera prueba de concepto para ayudar a identificar maquinas con mayor riesgo de fallo y priorizar tareas de mantenimiento.

## Dataset

Se utiliza el dataset `AI4I 2020 Predictive Maintenance Dataset`.

El dataset contiene:

- 10000 registros
- 14 columnas
- Variable objetivo: `Machine failure`

La variable objetivo indica:

- `0`: maquina sin fallo
- `1`: maquina con fallo

Variables utilizadas para el modelo principal:

- `Type`
- `Air temperature [K]`
- `Process temperature [K]`
- `Rotational speed [rpm]`
- `Torque [Nm]`
- `Tool wear [min]`

No se usan como predictoras las columnas `TWF`, `HDF`, `PWF`, `OSF` y `RNF`, ya que representan tipos concretos de fallo y podrian introducir fuga de informacion.

## Modelos probados

Se compararon varios modelos supervisados:

- Regresion Logistica
- Arbol de Decision
- Random Forest
- Gradient Boosting
- KNN

Tambien se probo un modelo no supervisado:

- Isolation Forest

El modelo no supervisado se uso como analisis complementario para detectar comportamientos anomalos.

## Modelo final

El mejor modelo general fue `Gradient Boosting` con umbral de decision ajustado a `0.3`.

Resultados para la clase fallo:

- Precision: 0.80
- Recall: 0.82
- F1-score: 0.81

Este modelo ofrece un buen equilibrio entre detectar fallos reales y evitar demasiadas falsas alarmas.

## Demo en Streamlit

El proyecto incluye una demo interactiva en Streamlit orientada a un tecnico de mantenimiento.

La aplicacion permite:

- Evaluar una maquina de forma individual.
- Usar casos de ejemplo.
- Calcular la probabilidad de fallo.
- Clasificar el riesgo como bajo, medio o alto.
- Mostrar señales operativas a revisar.
- Añadir maquinas a una tabla temporal de seguimiento.
- Descargar la tabla de maquinas evaluadas como CSV.
- Consultar metricas del modelo final.
- Ver una explicacion global mediante importancia de variables.

La tabla de maquinas evaluadas funciona como una simulacion temporal dentro de la app. En una version real, esta informacion deberia guardarse en una base de datos o conectarse a sistemas internos de la empresa.

## Ejecutar la app de Streamlit

Desde la raiz del proyecto, activar el entorno virtual:

```powershell
.\.venv\Scripts\Activate.ps1

Para ejecutarla:
streamlit run app_streamlit/app.py




## Estructura del proyecto

```text
predictive_maintenance_project/
│
├── app_streamlit/
│   ├── app.py
│   └── requirements.txt
│
├── data/
│   └── raw/
│       └── ai4i2020.csv
│
├── models/
│   └── final_model.pkl
│
├── notebooks/
│   ├── 01_datos.ipynb
│   ├── 02_modelado.ipynb
│   └── 03_conclusiones.ipynb
│
├── requirements.txt
├── .gitignore
└── README.md