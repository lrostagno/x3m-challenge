# x3m-challenge

## Pasos para probar la API:

1. Clonar el repositorio: 

    * `git clone https://github.com/lrostagno/x3m-challenge.git`


2. Moverse a la carpeta del proyecto con el comando `cd`

3. Crear el entorno virtual e instalar librerías:

   * `conda create --name x3m-challenge`
   * `conda activate x3m-challenge`
   * `conda install --file requirements.txt`

4. Descargar el modelo: `https://drive.google.com/file/d/1Qy87kJz9eIepvS3RIT2e93eI0YbXvVmW/view?usp=sharing`

5. Abrir el archivo `.env` en la carpeta `code` y reemplazar los valores de las variables `MODEL_PATH` por la ruta donde se encuentra el modelo, y `DATA_PATH` por la ruta donde se encuentra el dataset sobre el que se quieren realizar predicciones (se espera que tenga el mismo formato que el dataset del desafío).

6. Correr el archivo `code/app.py`:

    * `python code/app.py`


7. Abrir el archivo `code/test_API.py` y ejecutar las celdas del notebook.

El resultado que se obtendrá de la API es un json que posee, para cada instancia del dataset, el id del evento de ejecución de waterfall, el identificador de la instancia y la latencia predecida.

Se recomienda utilizar un tamaño de dataset de 2 a 20 waterfalls.

## Entrenamiento:
El código en el que se realizó el entrenamiento está en el archivo `code/training.py`
