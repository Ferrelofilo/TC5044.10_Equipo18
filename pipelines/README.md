# Pipelines

Este proyecto contiene una estructura modular para pipelines de procesamiento de datos, modelos y transformaciones. A continuación se detalla la estructura del proyecto y una breve descripción de los archivos y directorios principales.

## Estructura de pipelines

    pipelines/
                ├── utils/ 
                    ├── data_utils.py # Funciones auxiliares para la manipulación de datos 
                ├── models/ 
                    ├── simple_linear_cnn_multi_out_3.py # Implementación de un modelo CNN multisalida 
                ├── transformers/ 
                    ├── flare_column_transformer.py # Transformaciones específicas para columnas 
                └── src/
                    ├── etl.py # Extracción, transformación y carga de datos 
                    ├── preprocess.py # Funciones de preprocesamiento de datos 
                    ├── preprocess_v2.py # Segunda versión del preprocesamiento 
                    ├── train.py # Script de entrenamiento de


## Descripción de carpetas

- **pipelines/utils**: Contiene utilidades generales para el procesamiento y manejo de datos. El archivo `data_utils.py` incluye funciones auxiliares que son usadas en diferentes etapas del pipeline.

- **pipelines/models**: Aquí se definen los modelos de Machine Learning. El archivo `simple_linear_cnn_multi_out_3.py` contiene una implementación de un modelo de red neuronal convolucional simple con múltiples salidas.

- **pipelines/transformers**: Este módulo contiene transformadores que se aplican a las columnas de datos. El archivo `flare_column_transformer.py` incluye transformaciones específicas aplicadas a los datos.

- **pipelines/src**: Este módulo contiene scripts principales para la ejecución de los pipelines de ETL, preprocesamiento, entrenamiento y evaluación de modelos. Incluye varios scripts como `etl.py`, `train.py`, y `eval.py` que orquestan las distintas fases del flujo de trabajo.
