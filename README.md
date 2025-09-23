# TC02__Algoritmos_geneticos

## 🎨 Trabajo de arte evolutivo con algoritmos genéticos.

> Aproximación de una imagen objetivo mediante composición evolutiva de círculos con color y transparencia.

Este proyecto implementa un algoritmo genético que, generación tras generación, evoluciona una población de individuos (compuestos por círculos) para que la imagen renderizada se asemeje lo más posible a una imagen objetivo.

## 📦 Dependencias

Este proyecto requiere las siguientes librerías de Python, instalables con `pip`:

- **`numpy`**: Para operaciones numéricas eficientes con arrays (cálculo de fitness, mutación, vectores de cromosomas).
- **`Pillow` (PIL)**: Para cargar, manipular y guardar imágenes (carga de `target.png`, renderizado de círculos, guardado de snapshots).
- **`matplotlib`**: Para generar gráficas de evolución del fitness (máximo y promedio por generación).
- **`PyYAML`**: Para cargar configuraciones experimentales desde archivos `.yaml` (ajuste más sencillo de parámetros).

## 🚀 Ejecutar

Para ejecutar, abra una terminal en la raíz del proyecto, y ejecute el siguiente comando:

**`python -m experiments.run_experiment --config experiments/conf_a.yaml`**

Sustituyendo **`conf_a.yaml`** por **`conf_b.yaml`** o **`conf_c.yaml`** según requiera probar las diferentes configuraciones predeterminadas. 