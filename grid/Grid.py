import tkinter as tk

SQUARE_SIZE = 35


class GridSquare:
    def __init__(self, square: tk.Frame, label: tk.StringVar, row, col):
        self.square = square
        self.row = row
        self.col = col
        self.label = label

    def set_value(self, value):
        self.label.set(value)


class Grid:
    def __init__(self, parent_frame: tk.Frame, rows: int, cols: int):
        self.parent_frame = parent_frame
        self.rows = rows
        self.cols = cols
        self.matriz = []

        self._generar_widgets()

    def _generar_widgets(self):
        for row in range(self.rows):
            matriz_row = []
            for col in range(self.cols):
                square = tk.Frame(
                    self.parent_frame,
                    width=SQUARE_SIZE,
                    height=SQUARE_SIZE,
                    borderwidth=1,
                    highlightthickness=1,
                    highlightbackground="black",
                )
                square.grid(row=row, column=col)
                square.pack_propagate(False)

                lbl_string = tk.StringVar()
                lbl_string.set("0")
                lbl = tk.Label(square, textvariable=lbl_string)
                lbl.pack(expand=True)

                grid_square = GridSquare(square, lbl_string, row, col)

                # --- CAMBIOS CLAVE ---
                # 1. Vinculamos el evento al widget que está más arriba (el Label)
                lbl.bind('<B1-Motion>', self.on_square_drag)
                square.bind('<B1-Motion>', self.on_square_drag)
                # 2. Guardamos una referencia a nuestro objeto en el widget
                lbl.grid_square_ref = grid_square
                square.grid_square_ref = grid_square

                matriz_row.append(grid_square)
            self.matriz.append(matriz_row)

        # --- MÉTODO CORREGIDO ---

    def on_square_drag(self, event):
        """
        Manejador que encuentra dinámicamente el widget bajo el puntero del mouse.
        """
        # 1. Obtener las coordenadas GLOBALES del mouse desde el evento
        x_root = event.x_root
        y_root = event.y_root

        # 2. Preguntar al frame contenedor qué widget está en esas coordenadas
        #    winfo_containing() nos devuelve el widget que está actualmente bajo el puntero
        widget_actual = self.parent_frame.winfo_containing(x_root, y_root)

        # 3. Comprobar que hemos encontrado un widget y que es uno de los nuestros
        #    (usando la referencia que le añadimos antes)
        if widget_actual and hasattr(widget_actual, 'grid_square_ref'):
            s: GridSquare = widget_actual.grid_square_ref
            s.set_value("1")
            # Descomenta la siguiente línea para depurar y ver que funciona
            # print(f"Pintando casilla en (fila={s.row}, col={s.col})")