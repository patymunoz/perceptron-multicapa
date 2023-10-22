import pandas as pd
import numpy as np

def generar_sets_df(df, col_bool, prop, prop_interna_f, prop_interna_t):
    """
    Genera un conjunto de datos de entrenamiento y otro de prueba a partir de un dataframe, una columna booleana y una proporción entre entrenamiento y prueba y otra proporción
    para asegurar representación interna entre el 0 y 1.
    
    Parámetros:
    ------------
    df: 
        df con los datos a separar.
    col_bool: str
        Nombre de la columna booleana.
    prop: float
        Proporción de datos de entrenamiento.
    prop_interna_f: float
        Proporción de datos interna (false) para lograr representación entre la proporción de 0 y 1.
    prop_interna_t: float
        Proporción de datos interna (true) para lograr representación entre la proporción de 0 y 1.

    Regresa:
    --------
    train_df:
        df con los datos de entrenamiento.
    test_df:
        df con los datos de prueba.
    """
    
    # cantidad de entradas True y False
    n = df.shape[0]
    total_true = sum(df[col_bool])
    total_false = len(df) - total_true

    n_train = int(prop * n)
    n_test = n - n_train

    train_false = int(prop_interna_f*n_train)
    train_true = n_train - train_false

    test_false = int(prop_interna_f*n_test)
    test_true = n_test - test_false

    # seleccionar aleatoriamente las filas para los valores "false" (0)
    df_false = df[df[col_bool] == 0]
    df_train_false = df_false.sample(n=train_false, random_state=42)
    df_test_false = df_false.drop(df_train_false.index)

    # seleccionar aleatoriamente las filas para los valores 'true' (1)
    df_true = df[df[col_bool] == 1]
    df_train_true = df_true.sample(n=train_true, random_state=42)
    df_test_true = df_true.drop(df_train_true.index)

    # concatenar los conjuntos de entrenamiento y prueba
    train_df = pd.concat([df_train_true, df_train_false]).sample(frac=1, random_state=42)
    test_df = pd.concat([df_test_true, df_test_false]).sample(frac=1, random_state=42)
    
    return train_df, test_df

def generar_sets_arr(matriz, arr, prop, prop_interna_f, prop_interna_t):
    """
    Genera un conjunto de datos de entrenamiento y otro de prueba a partir de una matriz y una proporción 
    entre entrenamiento y prueba y otra proporción para asegurar representación interna entre el 0 y 1.
    
    Parámetros:
    ------------
    matriz: numpy array
        Matriz con los datos a separar.
    arr: numpy array
        Arreglo con los valores booleanos.
    prop: float
        Proporción de datos de entrenamiento.
    prop_interna_f: float
        Proporción de datos interna (false) para lograr representación entre la proporción de 0 y 1.
    prop_interna_t: float
        Proporción de datos interna (true) para lograr representación entre la proporción de 0 y 1.

    Regresa:
    --------
    train_subset:
        Matriz con los datos de entrenamiento.
    test_subset:
        Matriz con los datos de prueba.
    """
        
    n = len(arr)
    total_true = np.sum(arr)
    total_false = n - total_true

    n_train = int(prop * n)
    n_test = n - n_train

    train_false = int(prop_interna_f * n_train)
    train_true = n_train - train_false

    test_false = int(prop_interna_f * n_test)
    test_true = n_test - test_false

    # obtener índices de los valores "false" (0) y "true" (1)
    idx_false = np.where(arr == 0)[0]
    idx_true = np.where(arr == 1)[0]

    # seleccionar aleatoriamente los índices
    train_idx_false = np.random.choice(idx_false, train_false, replace=False)
    test_idx_false = np.setdiff1d(idx_false, train_idx_false)

    train_idx_true = np.random.choice(idx_true, train_true, replace=False)
    test_idx_true = np.setdiff1d(idx_true, train_idx_true)

    # obtener las filas de la matriz completa a partir de los índices
    train_subset_false = matriz[train_idx_false]
    test_subset_false = matriz[test_idx_false]

    train_subset_true = matriz[train_idx_true]
    test_subset_true = matriz[test_idx_true]

    # concatenar y mezclar
    train_subset = np.vstack([train_subset_true, train_subset_false])
    np.random.shuffle(train_subset)

    test_subset = np.vstack([test_subset_true, test_subset_false])
    np.random.shuffle(test_subset)
    
    return train_subset, test_subset