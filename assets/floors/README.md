# assets/floors (OPCIONAL)

Esta carpeta puede estar **vacia**. El simulador genera suelos procedurales
aleatorios por episodio sin necesidad de imagenes.

Si colocas aqui imagenes `.jpg` / `.png` de suelos reales (madera, baldosa,
marmol, etc.), el simulador las mezclara con las texturas procedurales con
probabilidad `--real-floor-prob` (default 0.3). Si la carpeta esta vacia, se
usa 100% procedural sin error.

Cada imagen debe ser idealmente cuadrada y tileable (por ejemplo 512x512 o
1024x1024).
