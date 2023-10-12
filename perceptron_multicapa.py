import numpy as np

# -------------------- # Preparación de los datos # ---------------------- #

# función de activación (el cerebro)
def func_sigmoidea(x):
    return 1 / (1 + np.exp(-x))

# -------------------- # función de entrenamiento # ---------------------- #
def proceso_entrenamiento(x, d, w_h, w_o, alfa, precision):
    """
    Genera los pesos del perceptrón multicapa.

    Parámetros:
    ----------
    x : np.ndarray
        Matriz de entradas.
    d : np.ndarray
        Matriz de salidas deseadas.
    w_h : np.ndarray
        Matriz de pesos de la capa oculta.
    w_o : np.ndarray
        Matriz de pesos de la capa de salida.
    alfa : float
        Tasa de aprendizaje.
    precision : float
        Precisión del error.
    
    Regresa:
    --------
    w_h : np.ndarray
        Matriz de pesos de la capa oculta.
    w_o : np.ndarray
        Matriz de pesos de la capa de salida.
    """
    # Procedimiento de entrenamiento
    E = float('inf')

    while E > precision:
        E = 0
        for j in range(Q):
            # proceso forward
            net_h = np.dot(w_h, x[j].T)
            y_h = func_sigmoidea(net_h)
            net_o = np.dot(w_o, y_h)
            y = func_sigmoidea(net_o)

            # proceso backward
            delta_o = ((d[j] - y) * y * (1 - y))
            delta_h = y_h * (1 - y_h) * np.dot(w_o.T, delta_o)
            # ajuste de los pesos (w's)
            Delta_w_o = alfa * np.outer(delta_o, y_h)
            Delta_w_h = alfa * np.outer(delta_h, x[j])

            w_o = w_o + Delta_w_o
            w_h = w_h + Delta_w_h

            # Acumulando el error
            E += np.linalg.norm(delta_o)

        print(f'error: {E}')

    return w_h, w_o 

# ------------ # función de prueba de funcionamiento # --------------- #
def proceso_funcionamiento(x, w_h, w_o):
    """
    Desarrolla el proceso de funcionamiento del perceptrón multicapa, incluyendo nuevas entradas con
    los pesos ya entrenados.

    Parámetros:
    ----------
    x : np.ndarray
        Matriz de entradas.
    w_h : np.ndarray
        Matriz de pesos de la capa oculta.
    w_o : np.ndarray
        Matriz de pesos de la capa de salida.
    
    Regresa:
    --------
    y_round : np.ndarray
        Matriz de salidas redondeadas.
    """
    y = np.zeros((Q, M))
    for j in range(Q):
        net_h = np.dot(w_h, x[j].T)
        y_h = func_sigmoidea(net_h)
        net_o = np.dot(w_o, y_h)
        y[j] = func_sigmoidea(net_o)
        y_round = np.round(y, 0).astype(int)

    return y_round