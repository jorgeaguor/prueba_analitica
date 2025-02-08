import logging
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

class modelo():
    def __enter__(self):
        return self
    
    def __init__(self, teardown=False):

        ruta_log = 'Log/Eventos.log'
        logging.basicConfig(filename=ruta_log, level='INFO',format='%(asctime)s - %(levelname)s - %(message)s')
        self.teardown = teardown
        super(modelo, self).__init__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.teardown:
            self.quit()


    def leer_archivos(self):
        try:
            # Lista de clientes y obligaciones a calificar en 2024/01

            clientes_obligaciones = pd.read_csv('./raw/prueba_op_base_pivot_var_rpta_alt_enmascarado_oot.csv')
            print(f'Tabla de obligaciones a calificar en 2024/01 leida')
            logging.info(f'Lectura de dataframe obligaciones clientes exitosa: {clientes_obligaciones.shape[0]} filas obtenidas')


            # Información de variable objetivo para train

            clientes_obligaciones_train = pd.read_csv('./raw/prueba_op_base_pivot_var_rpta_alt_enmascarado_trtest.csv')
            # Eliminar duplicados en columnas 'vlr_obligacion', 'pago_cuota', 'valor_cuota_mes', 'rpc'
            clientes_obligaciones_train = clientes_obligaciones_train.dropna(subset=['vlr_obligacion', 'pago_cuota', 'valor_cuota_mes', 'rpc'])
            clientes_obligaciones_train['porc_pago_mes'] = clientes_obligaciones_train['porc_pago_mes'].fillna(0)
            clientes_obligaciones_train['cant_acuerdo'] = clientes_obligaciones_train['cant_acuerdo'].fillna(0)
            clientes_obligaciones_train['cant_gestiones'] = clientes_obligaciones_train['cant_gestiones'].fillna(0)
            # Dejar todos los valores de la columna producto en mayúscula y homologar valores
            clientes_obligaciones_train['producto'] = clientes_obligaciones_train['producto'].str.upper()
            clientes_obligaciones_train['producto'] = clientes_obligaciones_train['producto'].replace('TARJETAS DE CREDITO', 'TARJETA DE CREDITO')
            clientes_obligaciones_train['segmento'] = clientes_obligaciones_train['segmento'].str.upper()
            clientes_obligaciones_train['banca'] = clientes_obligaciones_train['banca'].str.upper()
            print(f'Tabla de información variable objetivo train leida')
            logging.info(f'Lectura de dataframe obligaciones clientes entrenamiento exitosa: {clientes_obligaciones_train.shape[0]} filas obtenidas')


            # Información histórica pago de obligaciones

            pagos_obligaciones = pd.read_csv('./raw/prueba_op_maestra_cuotas_pagos_mes_hist_enmascarado_completa.csv')
            # Eliminar filas donde valor_cuota_mes y pago_total sean iguales a 0
            pagos_obligaciones = pagos_obligaciones[~((pagos_obligaciones['valor_cuota_mes'] == 0) & (pagos_obligaciones['pago_total'] == 0))]
            # Unificar los nombres de los productos
            pagos_obligaciones['producto'] = pagos_obligaciones['producto'].str.upper()
            pagos_obligaciones['producto'] = pagos_obligaciones['producto'].replace('TARJETA DE CRÉDITO', 'TARJETA DE CREDITO')

            # Obtener periodicidad de obligaciones

            fechas_por_obligacion = pagos_obligaciones.drop_duplicates(subset=['nit_enmascarado', 'num_oblig_enmascarado', 'fecha_corte'])
            fechas_por_obligacion = fechas_por_obligacion.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['fecha_corte'].unique().apply(sorted).reset_index()

            def calcular_frecuencia_numerica(fechas):
                if len(fechas) < 2:
                    return np.nan  # Si hay menos de 2 fechas, no se puede calcular una frecuencia
                
                # Extraer año y mes de la fecha en formato AAAAMMDD
                anios = [f // 10000 for f in fechas]   # Extraer AAAA
                meses = [(f // 100) % 100 for f in fechas]  # Extraer MM
                
                # Convertir (Año, Mes) a un número de meses totales desde el año 0
                meses_totales = [a * 12 + m for a, m in zip(anios, meses)]
                
                # Calcular diferencias en meses
                diferencias = [meses_totales[i] - meses_totales[i-1] for i in range(1, len(meses_totales))]
                
                # Si todas las diferencias son iguales, devolver la frecuencia, si no, indicar 'Variable'
                return diferencias[0] if all(x == diferencias[0] for x in diferencias) else 'Variable'

            print(f'Calculando periodicidad de la información para tabla historica de obligaciones')

            fechas_por_obligacion['frecuencia_meses'] = fechas_por_obligacion['fecha_corte'].apply(calcular_frecuencia_numerica)
            fechas_por_obligacion = fechas_por_obligacion[fechas_por_obligacion['frecuencia_meses']==1]

            # Filtrar tabla inicial con clientes y productos que tengan información mensual
            pagos_obligaciones = pagos_obligaciones.merge(fechas_por_obligacion, on=['nit_enmascarado', 'num_oblig_enmascarado'])

            print(f'Informacion tabla historica de obligaciones filtrada por nits y obligaciones con data mensual')
            print(f'Tabla de informacion historica de obligaciones leida')
            logging.info(f'Lectura de dataframe informacion historica obligaciones exitosa: {pagos_obligaciones.shape[0]} filas obtenidas')


            # Información demográfica del cliente

            demografica_cliente = pd.read_csv('./raw/prueba_op_master_customer_data_enmascarado_completa.csv')
            # Obtener última ingestión por cliente
            demografica_cliente = demografica_cliente.sort_values(by=['nit_enmascarado', 'year', 'month', 'ingestion_day'], ascending=[True, False, False, False])
            demografica_cliente = demografica_cliente.drop_duplicates(subset=['nit_enmascarado'], keep='first')

            print(f'Tabla de informacion demografica leida')
            logging.info(f'Lectura de dataframe informacion demografica exitosa: {demografica_cliente.shape[0]} filas obtenidas')


            # Información de probabilidades de modelos desarrollados

            probabilidades_modelos = pd.read_csv('./raw/prueba_op_probabilidad_oblig_base_hist_enmascarado_completa.csv')

            print(f'Tabla de informacion modelos desarrollados leida')
            logging.info(f'Lectura de dataframe modelos desarrollados exitosa: {probabilidades_modelos.shape[0]} filas obtenidas')


            logging.info(f'Toda la informacion leida exitosamente')
            print(f'Toda la informacion leida exitosamente: clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos')

            return clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos
        
        except:

            logging.info('Lectura de archivos fallida')


    
    def filtrar_columnas_a_usar(self, clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos):

        try:

            clientes_obligaciones = clientes_obligaciones
            logging.info(f'Columnas seleccionadas tabla de obligaciones a calificar en 2024/01')
            print(f'Columnas seleccionadas tabla de obligaciones a calificar en 2024/01')

            clientes_obligaciones_train = clientes_obligaciones_train[['nit_enmascarado',
                                                                       'num_oblig_orig_enmascarado',
                                                                       'num_oblig_enmascarado',
                                                                       'producto',
                                                                       'fecha_var_rpta_alt',
                                                                       'var_rpta_alt',
                                                                       'banca',
                                                                       'segmento',
                                                                       'rpc',
                                                                       'cant_alter_posibles']]

            logging.info(f'Columnas seleccionadas tabla de información variable objetivo train')
            print(f'Columnas seleccionadas tabla de información variable objetivo train')

            pagos_obligaciones = pagos_obligaciones[['nit_enmascarado',
                                                     'num_oblig_enmascarado',
                                                     'fecha_corte_x',
                                                     'valor_cuota_mes',
                                                     'pago_total',
                                                     'porc_pago',
                                                     'ajustes_banco']]
            
            pagos_obligaciones = pagos_obligaciones.rename(columns={'fecha_corte_x': 'fecha_corte'})
            
            logging.info(f'Columnas seleccionadas tabla historica de obligaciones')
            print(f'Columnas seleccionadas tabla historica de obligaciones')
            
            demografica_cliente = demografica_cliente[['nit_enmascarado',
                                                       'cod_tipo_doc',
                                                       'tipo_cli',
                                                       'ctrl_terc',
                                                       'edad_cli',
                                                       'nivel_academico',
                                                       'segm',
                                                       'subsegm',
                                                       'nicho',
                                                       'region_of']]
            
            logging.info(f'Columnas seleccionadas tabla demografica clientes')
            print(f'Columnas seleccionadas tabla demografica clientes')
            
            probabilidades_modelos = probabilidades_modelos

            logging.info(f'Columnas seleccionadas tabla probabilidades modelos')
            print(f'Columnas seleccionadas tabla probabilidades modelos')

            return clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos

        except:
            
            logging.info('Seleccion de informacion fallida')


    
    def unir_tablas(self, clientes_obligaciones, clientes_obligaciones_train, pagos_obligaciones, demografica_cliente, probabilidades_modelos):

        try:
            
            # Data de entrenamiento

            # Dejar únicamente el registro más actual de nit-obligación
            clientes_obligaciones_train = clientes_obligaciones_train.loc[clientes_obligaciones_train.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['fecha_var_rpta_alt'].idxmax()]

            # Unir clientes_obligaciones_train y pagos_obligaciones
            data_train = pd.merge(clientes_obligaciones_train, pagos_obligaciones, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='inner')
            logging.info(f'Unión entre obligaciones clientes train y pagos obligaciones exitosa')
            print(f'Unión entre obligaciones clientes train y pagos obligaciones exitosa')

            # Unir con demografica_cliente
            data_train = pd.merge(data_train, demografica_cliente, on=['nit_enmascarado'], how='left')
            logging.info(f'Unión con demografica cliente exitosa')
            print(f'Unión con demografica cliente exitosa')

            # Dejar únicamente el registro más actual de nit-obligación
            probabilidades_modelos = probabilidades_modelos.loc[probabilidades_modelos.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['fecha_corte'].idxmax()]

            # Unir con probabilidades_modelos
            data_train = pd.merge(data_train, probabilidades_modelos, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='inner')
            logging.info(f'Unión con probabilidades exitosa')
            print(f'Unión con probabilidades exitosa')
            logging.info(f'Unión para data train exitosa')
            print(f'Unión para data train exitosa')

            # Data para predecir con el modelo
           
            # Unir clientes_obligaciones y clientes_obligaciones_train
            data_predict = pd.merge(clientes_obligaciones, clientes_obligaciones_train, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='left')
            
            logging.info(f'Unión entre obligaciones clientes y obligaciones clientes train exitosa')
            print(f'Unión entre obligaciones clientes y obligaciones clientes train exitosa')
            
            # Sacar el último registro de pagos por combinación nit - obligacion
            pagos_obligaciones_max_id = pagos_obligaciones.loc[pagos_obligaciones.groupby(['nit_enmascarado', 'num_oblig_enmascarado'])['fecha_corte'].idxmax()]

            # Unir clientes_obligaciones y pagos_obligaciones
            data_predict = pd.merge(data_predict, pagos_obligaciones_max_id, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='left')
            logging.info(f'Unión con pagos obligaciones exitosa')
            print(f'Unión con pagos obligaciones exitosa')

            # Unir con demografica_cliente
            data_predict = pd.merge(data_predict, demografica_cliente, on=['nit_enmascarado'], how='left')
            logging.info(f'Unión con demografica cliente exitosa')
            print(f'Unión con demografica cliente exitosa')

            # Unir con probabilidades_modelos
            data_predict = pd.merge(data_predict, probabilidades_modelos, on=['nit_enmascarado', 'num_oblig_enmascarado'], how='left')
            logging.info(f'Unión con probabilidades exitosa')
            print(f'Unión con probabilidades exitosa')

            logging.info(f'Unión para data a predecir exitosa')
            print(f'Unión para data a predecir exitosa')

            return data_train, data_predict

        except:

            logging.info('Consolidacion de informacion fallida')

    def consolidar_tablas(self, data_train, data_predict):

        try:
            
            # Generar columna de ID
            data_train['ID'] = data_train['nit_enmascarado'].astype(str) + '#' + data_train['num_oblig_orig_enmascarado'].astype(str) + '#' + data_train['num_oblig_enmascarado'].astype(str)

            # Generar rangos para las probabilidades
            data_train['prob_pago_siguiente_mes'] = pd.cut(data_train['prob_propension'], 
                         bins=[0, 0.4, 0.7, 1], 
                         labels=['Baja', 'Media', 'Alta'], 
                         include_lowest=True)
            data_train['prob_mora_siguiente_mes'] = pd.cut(data_train['prob_alrt_temprana'], 
                         bins=[0, 0.4, 0.7, 1], 
                         labels=['Baja', 'Media', 'Alta'], 
                         include_lowest=True)
            data_train['prob_ponerse_al_dia'] = pd.cut(data_train['prob_auto_cura'], 
                         bins=[0, 0.4, 0.7, 1], 
                         labels=['Baja', 'Media', 'Alta'], 
                         include_lowest=True)
            
            # Generar clasificaciones para los porcentajes de pago
            data_train['marca_pago'] = data_train['porc_pago'].apply(lambda x: 'IGUAL' if x == 100 else 'PAGO_MENOS' if x < 100 else 'PAGO_MAS')

            # Generar clasificaciones para edades
            data_train['rangos_edades'] = data_train['edad_cli'].apply(lambda x: 
                                   'MENOR' if x < 18 else 
                                   'AULTO_MAYOR' if x > 40 else 
                                   'ADULTO_JOVEN')
            
            data_train.drop(columns=['nit_enmascarado','num_oblig_orig_enmascarado','num_oblig_enmascarado', 'fecha_var_rpta_alt',
                                     'fecha_corte_x', 'valor_cuota_mes', 'pago_total', 'porc_pago', 'nivel_academico', 'segm', 'nicho',
                                     'fecha_corte_y', 'lote', 'prob_propension', 'prob_alrt_temprana', 'prob_auto_cura', 'edad_cli'], inplace=True)

            logging.info(f'Consolidacion data train exitosa')
            print(f'Consolidacion data train exitosa')

            # Generar columna de ID
            data_predict['ID'] = data_predict['nit_enmascarado'].astype(str) + '#' + data_predict['num_oblig_orig_enmascarado_x'].astype(str) + '#' + data_predict['num_oblig_enmascarado'].astype(str)

            # Generar rangos para las probabilidades
            data_predict['prob_pago_siguiente_mes'] = pd.cut(data_predict['prob_propension'], 
                         bins=[0, 0.4, 0.7, 1], 
                         labels=['Baja', 'Media', 'Alta'], 
                         include_lowest=True)
            data_predict['prob_mora_siguiente_mes'] = pd.cut(data_predict['prob_alrt_temprana'], 
                         bins=[0, 0.4, 0.7, 1], 
                         labels=['Baja', 'Media', 'Alta'], 
                         include_lowest=True)
            data_predict['prob_ponerse_al_dia'] = pd.cut(data_predict['prob_auto_cura'], 
                         bins=[0, 0.4, 0.7, 1], 
                         labels=['Baja', 'Media', 'Alta'], 
                         include_lowest=True)

            # Generar clasificaciones para los porcentajes de pago
            data_predict['marca_pago'] = data_predict['porc_pago'].apply(lambda x: 'IGUAL' if x == 100 else 'PAGO_MENOS' if x < 100 else 'PAGO_MAS')

            # Generar clasificaciones para edades
            data_predict['rangos_edades'] = data_predict['edad_cli'].apply(lambda x: 
                                   'MENOR' if x < 18 else 
                                   'AULTO_MAYOR' if x > 40 else 
                                   'ADULTO_JOVEN')

            data_predict.drop(columns=['nit_enmascarado','num_oblig_orig_enmascarado_x','num_oblig_enmascarado', 'fecha_var_rpta_alt_x', 'var_rpta_alt',
                                        'num_oblig_orig_enmascarado_y', 'fecha_var_rpta_alt_y', 'fecha_corte_x', 'valor_cuota_mes', 'pago_total',
                                        'porc_pago', 'nivel_academico', 'segm', 'nicho', 'fecha_corte_y', 'lote', 'prob_propension',
                                        'prob_alrt_temprana', 'prob_auto_cura', 'edad_cli'], inplace=True)

            return data_train, data_predict

        except:

            logging.info(f'Consolidacion data train fallida')


    def preparacion_dataset_entrenamiento(self, data_train):

        try:
            
            # Leer dataframe
            df =  data_train.copy()

            # Asignar variable y
            y = df['var_rpta_alt']

            # Asignar variable X
            X = df.copy()
            X.pop('var_rpta_alt')
            X.pop('ID')

            # Partir los datos, 20% de los datos para test y 80% para train
            # random state para que se guarde la semilla
            (X_train, X_test, y_train, y_test) = train_test_split(
                X,
                y,
                test_size = 0.2,
                random_state = 123,
            )

            logging.info(f'X_train, X_test, y_train, y_test generados exitosamente')
            print(f'Consolidacion X_train, X_test, y_train, y_test generados exitosamente exitosa')

            return X_train, X_test, y_train, y_test

        except:

            logging.info(f'Generacion X_train, X_test, y_train, y_test fallida')

        
    def construccion_pipeline(self, X_train, y_train):

        try:

            # Crear pipeline con estimador OneHotEncoder y un estimador
            # LogisticRegression con una regularización Cs=10
            # CV = 10 va a testear 10 valores diferentes para la regresión logística
            # y va a escoger el mejor

            pipeline = Pipeline(
                steps=[
                    ("onehotencoder", OneHotEncoder(handle_unknown='ignore')),
                    ("logisticregression", LogisticRegressionCV(Cs = 10)),
                ],
            )

            # Entrenar el pipeline
            
            pipeline.fit(X_train, y_train)

            logging.info(f'Construccion de pipeline exitoso')
            print(f'Construccion de pipeline exitoso')

            return pipeline

        except:

            logging.info(f'Construccion de pipeline fallido')
            print(f'Construccion de pipeline fallido')


    def evaluacion_pipeline(self, X_train, X_test, y_train, y_test, pipeline):

        try:

            # Comparar y verdadera y y predecida (train)
            y_true_train = y_train
            y_pred_train = pipeline.predict(X_train)

            # Calcular F1 score
            f1_train = f1_score(y_true_train, y_pred_train)

            print('F1 Score train: ', f1_train)
            logging.info(f'F1 Score train: {f1_train}')

            # Comparar y verdadera y y predecida (test)
            y_true_test = y_test
            y_pred_test = pipeline.predict(X_test)

            # Calcular F1 score
            f1_test = f1_score(y_true_test, y_pred_test)

            print('F1 Score test: ', f1_test)
            logging.info(f'F1 Score test: {f1_test}')

            logging.info(f'Evaluacion pipeline finalizada')
            print(f'Evaluacion pipeline finalizada')

        except:

            logging.info(f'Evaluacion pipeline interrumpida')
            print(f'Evaluacion pipeline interrumpida')

    
    def pipeline_prediccion(self, pipeline, data_predict):

        try:
            
            # Hacer la predicción
            X = data_predict.copy()

            # Partir los datos, sólo se toma un 30%
            #X_pred = X.sample(frac=0.3, replace=True, random_state=42)
            X_pred = X.copy()

            X_pred_1 = X_pred.drop(columns=['ID'])

            y_pred = pipeline.predict(X_pred_1)

            logging.info(f'Evaluacion pipeline finalizada')
            print(f'Evaluacion pipeline finalizada')

            # Asignar la variable resultado
            resultado = X_pred.copy()
            resultado['var_rpta_alt'] = y_pred

            # Tomar únicamente las dos columnas para el archivo de sumisión
            resultado = resultado[['ID', 'var_rpta_alt']]

            # Fecha actual
            hoy = datetime.today()

            # Año, mes y día
            anio = hoy.year
            mes = hoy.month
            dia = hoy.day

            # Exportar el resultado
            resultado.to_csv(f'./runs/respuesta_corrida_{anio}_{mes}_{dia}.txt', encoding='utf-8', sep=' ', index=False)
            logging.info(f'Evaluacion pipeline exitosa')
            logging.info(f'Generación de archivo respuesta_corrida_{anio}_{mes}_{dia}.txt exitosa')

        except:

            logging.info(f'Evaluacion pipeline interrumpida')
            print(f'Evaluacion pipeline interrumpida')



