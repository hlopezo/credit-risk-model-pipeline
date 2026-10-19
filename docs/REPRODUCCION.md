# Reproducción integral verificada desde la fase 1.0

Esta verificación sustituye el alcance incompleto de la primera entrega: ahora se comienza con la fuente sin SEGMENTO y se ejecuta literalmente el notebook 1.0 antes de las fases posteriores.
Entorno nuevo: outputs/clean_phase1_env, Python 3.14.4, sin system-site-packages. Se instaló requirements-lock y pip check no detectó dependencias rotas.
Comando ejecutado: `outputs\clean_phase1_env\Scripts\python.exe -X utf8 scripts/run_pipeline.py`. Orden completo: 1 → 2 → 3 → 4 → 5 → 5.1 → MLflow → pruebas → validación.

## Partición original reproducida

Fuente real: data/raw/base_credit_risk_202510.csv, separador punto y coma, 50,620 filas y 47 columnas sin SEGMENTO. Se retiró la entrada segmentada del repositorio versionable.
Código de partición: OOT 202508/202509/202510; desarrollo <=202507; TEST=30%, semilla 123, estratificación indicador_default. Las celdas originales de fase 1 se compararon y son idénticas; solo se añaden bootstrap, importación de presentación y validación final.
SHA256 de la fuente: `d8278614e2e822d871b098143317351f7ad5a8948e2970a87c348ea01eaba514`.
SHA256 del CSV generado y del original: `77257d30bc7e475d2219724ef99faf4cb8f75aed5ebd6524962caff4ecda7852`.
La igualdad byte a byte se complementa con validaciones de columnas, tipos, índices, IDs, target, filas, clientes y períodos por segmento. No se usa un CSV precalculado como entrada.

| Muestra | Registros | Clientes | Defaults | Períodos |
| --- | ---: | ---: | ---: | --- |
| TRAIN | 28221 | 28221 | 3001 | 202408–202507 |
| TEST | 12095 | 12095 | 1286 | 202408–202507 |
| OOT | 10304 | 10304 | 1131 | 202508–202510 |

## Pipeline completo

| Comprobación | Resultado |
| --- | --- |
| Variables después de univariado/PSI | 29 |
| B.3 aprobadas / rechazadas | 24 / 5 |
| Correlación | 11 |
| Modelo final | 9 |
| Cortes y WoE | Coinciden |
| Coeficientes e intercepto | Coinciden; JSON idéntico al original |
| Métricas | CSV idéntico al original |
| Pruebas | 226, sin fallos ni errores |

| Muestra | AUC | Gini | KS |
| --- | ---: | ---: | ---: |
| TRAIN | 0.75076916 | 0.50153832 | 0.36714295 |
| TEST | 0.75604869 | 0.51209737 | 0.37099980 |
| OOT | 0.76784152 | 0.53568304 | 0.40615440 |

El CSV completo de métricas y el JSON de coeficientes tienen el mismo SHA256 que sus originales. La diferencia de ~4.9e-16 frente a la referencia compacta se debe a su serialización con 15 decimales, no a cambios del modelo. Los hashes de procedencia y tiempos del entorno nuevo pueden cambiar; no se exigen pickles ni timestamps idénticos.

## Independencia del proyecto de origen

La ejecución usa exclusivamente la fuente y código contenidos en este repositorio. El guard de I/O bloquea el acceso al proyecto de origen en el proceso principal y subprocesos Python; sus registros no contienen intentos de acceso al origen. Los originales se leyeron solo para preparación de referencias y auditoría de integridad antes/después de ejecutar, nunca como entrada durante la reproducción.
Se verificaron 28 archivos originales de notebooks, módulos y datos sin cambios. Hay 9 registros de procesos finalizados con el bloqueo activo y cero intentos denegados.

## MLflow nuevo

UI: http://127.0.0.1:5002. SQLite nuevo: outputs/workspace_from_raw/MLFLOW_LOCAL/mlflow.db. El run univariado contiene además la evidencia agregada de partición 1.0. El modelo se recuperó desde Model Registry y reprodujo probabilidades sintéticas a tolerancia 1e-12.

| Run | ID |
| --- | --- |
| 01_UNIVARIADO_PSI | `8c94e3fcbfdf407c9b9964598b456999` |
| 02_BIVARIADO_WOE | `d15a2defa7b24148a06bd1bce32e0a06` |
| 03_MULTIVARIADO_CORRELACION | `5080b301ffe641d9bcc14839040350d2` |
| 04_MODELO_LOGISTICO_FINAL | `372336cb3df4476290a703cf6890800b` |

## Evidencia local y versionable

- docs/reproduction_verified.json: comprobaciones estructuradas de esta reproducción integral.
- configs/partition_reference.json: contrato de la partición original, sin filas precalculadas.
- outputs/workspace_from_raw/1.SEGMENTACION/evidencias/particiones_verificadas.json: comparación real por muestra.
- outputs/executed_notebooks/phase1_split_calls.json: parámetros observados en la llamada real a sklearn.
- outputs/executed_notebooks/: seis notebooks ejecutados sin errores de salida.
- outputs/access_audit/: control de acceso al proyecto original.
- outputs/tests.json y tests.txt: 226 pruebas aprobadas.
Git sigue sin commits ni push. Las ejecuciones automatizadas son Python secuencial con captura de outputs; no se atribuyen a una sesión interactiva de kernel Jupyter.