from improve.improve import modelo
import pandas as pd

with modelo() as bot:
    clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos = bot.leer_archivos()
    clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos = bot.filtrar_columnas_a_usar(clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos)
    data_train, data_predict = bot.unir_tablas(clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos)
    data_train, data_predict = bot.consolidar_tablas(data_train, data_predict)
    X_train, X_test, y_train, y_test = bot.preparacion_dataset_entrenamiento(data_train)
    pipeline = bot.construccion_pipeline(X_train, y_train)
    bot.evaluacion_pipeline(X_train, X_test, y_train, y_test, pipeline)
    bot.pipeline_prediccion(pipeline, data_predict)

