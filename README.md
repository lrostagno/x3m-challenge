# x3m-challenge

## Pasos para probar la API:

1. git clone https://github.com/lrostagno/x3m-challenge.git

2. Moverse a la carpeta del proyecto con el comando `cd`

3. Crear el entorno virtual e instalar librerías:

   * `conda create --name x3m-challenge`
   * `conda activate x3m-challenge`
   * `conda install --file requirements.txt`

4. Descargar el modelo: https:...asdf....

5. Abrir el archivo `.env` en la carpeta `code` y reemplazar los valores de las variables `MODEL_PATH` por la ruta donde se encuentra el modelo, y `DATA_PATH` por la ruta donde se encuentra el dataset sobre el que se quieren realizar predicciones (se espera que tenga el mismo formato que el dataset del desafío).

6. Correr el archivo `code/app.py`:

    * `python code/app.py`


7. Abrir el archivo `code/test_API.py` y ejecutar las celdas del notebook