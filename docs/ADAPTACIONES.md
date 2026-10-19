# Adaptaciones de empaquetado

La reproducción empieza en el notebook 1.0, que se ejecuta con su código original de carga y segmentación. Se agrega bootstrap portable, la importación faltante FuncFormatter y una celda final de validación; no se cambian las celdas originales ni el split.

Se conserva la organización src/scripts/tests/notebooks y el adaptador de compatibilidad de rutas. Ahora solo se copia la fuente sin segmentar a 1.base del workspace; 2.OUTPUT se genera por ejecución. La fase 2 exige un contrato de fase 1 válida. El workspace y backend SQLite de esta prueba son nuevos y están separados de los de la entrega anterior.

Las funciones estadísticas posteriores permanecen sin cambios. La referencia WoE de selección 4.0 se serializa relativa; las demás adaptaciones son I/O/configuración/captura de outputs. Los antecedentes B.2 y resultados esperados se usan únicamente para auditoría y comparación.
