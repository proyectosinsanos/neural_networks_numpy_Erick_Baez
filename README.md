
### Red Neuronal para Clasificación de Datos

### Descripción
Este proyecto implementa una red neuronal artificial (ANN) desde cero utilizando NumPy. La red neuronal es capaz de clasificar datos generados mediante distribuciones gaussianas. Se emplea una arquitectura de tres capas ocultas con funciones de activación ReLU y sigmoide.

### Requisitos
Para ejecutar este proyecto, se necesitan las siguientes bibliotecas:
- `numpy`: Para operaciones matemáticas y manejo de datos.
- `matplotlib`: Para la visualización de los datos y los resultados.
- `sklearn.datasets`: Para la generación de datos de prueba.

Puedes instalarlas con el siguiente comando:
```sh
pip install numpy matplotlib scikit-learn
```

### Estructura del Código
1. **Generación de datos**: Se crean puntos de datos con la función `make_gaussian_quantiles`.
2. **Definición de funciones de activación**:
   - `sigmoid(x)`: Función de activación sigmoide.
   - `relu(x)`: Función de activación ReLU.
3. **Función de pérdida**:
   - `mse(y, y_hat)`: Error cuadrático medio.
4. **Inicialización de parámetros**:
   - `initialize_parameters_deep(layers_dims)`: Inicializa pesos y sesgos aleatoriamente.
5. **Entrenamiento de la red neuronal**:
   - `train(x_data, y_data, learning_rate, params, training=True)`: Realiza la propagación hacia adelante y, si `training=True`, actualiza los pesos mediante backpropagation.
6. **Ejecución principal**:
   - Se entrena la red neuronal durante 50,000 iteraciones.
   - Se visualizan los datos de entrenamiento y los resultados de la clasificación.

### Ejecución del Proyecto
Para ejecutar el código, simplemente corre el siguiente comando en la terminal:
```sh
python main.py
```


### Resultados
- Se genera un conjunto de datos de clasificación.
- La red neuronal aprende a separar las clases.
- Se muestran gráficos con los datos de entrenamiento y prueba.
