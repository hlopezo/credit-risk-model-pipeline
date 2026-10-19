# Auditoría de la omisión de fase 1.0

La primera entrega eligió como entrada el CSV ya segmentado que consumía la fase 2.0. El runtime copiaba ese CSV; no existía un script que reconstruyera la partición. Por tanto, la reproducción anterior sí verificó las fases 2.0–5.1, pero no era integral desde la base original. La omisión fue una decisión de alcance incorrecta, no una imposibilidad técnica.

El notebook 1.0 revela una partición mixta: OOT para 202508/202509/202510, desarrollo hasta 202507 y train_test_split únicamente dentro de desarrollo con test_size=0.30, random_state=123 y estratificación por indicador_default. Se conserva literalmente ese código; no se sustituye por un split aleatorio global.

Se incorporó la fuente original separada por punto y coma, de 50,620 registros y 47 columnas sin SEGMENTO. Se retiró el CSV precalculado de data/raw. El orquestador ahora ejecuta 1.0 primero y valida su archivo generado antes de admitir fase 2. Las referencias contienen hashes y resúmenes, no datos para fabricar segmentos.

El notebook original no importaba FuncFormatter pese a utilizarlo en una figura. Se añade únicamente esa importación al bootstrap de presentación. Se conservan todas sus importaciones originales, declarando feature-engine y xgboost en requirements; no se alteran selección, semilla, estratificación, target ni asignación por índices.

La validación compara archivo completo, columnas/tipos, filas por muestra, períodos, clientes, defaults, índices, identificadores y hashes de registros. Una frontera I/O impide leer el proyecto original durante la ejecución y sus subprocesos. El código original no se edita.
