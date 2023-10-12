# perceptron-multicapa

# Introducción


# Instalación del repositorio

Para poder llevar a cabo los procesamientos en el notebook "07_perceptron_multicapa_aplicacion.ipynb" es necesario instalar las dependencias que se encuentran en el archivo *requirements.in*.

1. En la carpeta donde se localiza este repositorio, crea un ambiente virtual con **Python 3.10** y actívalo de la siguiente manera:
    
    - con `pipenv`:
    ```bash
    pipenv shell --python 3.10
    ```
    - con `conda`:
    ```bash
    conda create --name nombre_del_ambiente python==3.10
    conda activate nombre_del_ambiente
    ```

2. Luego te activar el ambiente virtual y antes de sincronizar las dependencias que contiene el archivo *requirements*, asegúrate de instalar `pip-tools` dentro del ambiente virtual:
    
    - con `pip`: 
    ```bash
    pip install pip-tools==1.8.0
    ```
    - con `conda`:
    ```bash
    conda install -c conda-forge pip-tools
    ```

3. Mediante pip-sync instala las dependencias listadas en el archivo requirements.in:
    ```bash
    pip-sync requirements.in
    ```

4. Puedes asegurarte que la instalación de dependencias fue exitosa ejecutando el siguiente comando:
    ```bash
    pip list
    ```

    o 
    ```bash
    conda list
    ```
    dependiendo de si usaste `pip` o `conda` para instalar las dependencias.