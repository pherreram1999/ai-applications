import tkinter as tk
from Grid import Grid

def main():
    w = tk.Tk()
    w.title("Laberinto")

    grid_frame = tk.Frame(w)
    grid_frame.pack(padx=10, pady=10) # autocolocacion

    grid = Grid(grid_frame, 20, 20)

    w.mainloop()

if __name__ == '__main__':
    main()