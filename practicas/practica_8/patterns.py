import numpy as np
import matplotlib.pyplot as plt


#PATRON CUADRADO 20X20
patron_cuadrado = np.zeros((20, 20), dtype=int)
inicio = 3
fin = 17
patron_cuadrado[inicio:inicio+1, inicio:fin] = 1
patron_cuadrado[fin-1:fin, inicio:fin] = 1
patron_cuadrado[inicio:fin, inicio:inicio+1] = 1
patron_cuadrado[inicio:fin, fin-1:fin] = 1


#PATRON CIRCULO 20x20
patron_circulo = np.zeros((20, 20), dtype=int)
patron_circulo[3, 8:12] = 1   
patron_circulo[16, 8:12] = 1  
patron_circulo[8:12, 3] = 1   
patron_circulo[8:12, 16] = 1  
patron_circulo[4, 6:8] = 1   
patron_circulo[5, 5] = 1
patron_circulo[6:8, 4] = 1   
patron_circulo[4, 12:14] = 1
patron_circulo[5, 14] = 1
patron_circulo[6:8, 15] = 1
patron_circulo[15, 6:8] = 1
patron_circulo[14, 5] = 1
patron_circulo[12:14, 4] = 1
patron_circulo[15, 12:14] = 1
patron_circulo[14, 14] = 1
patron_circulo[12:14, 15] = 1


#PATRON TRIANGULO 20x20
patron_triangulo = np.zeros((20, 20), dtype=int)
patron_triangulo[16, 4:17] = 1
for fila in range(4, 16):
    progreso = (fila - 4) / (16 - 4)
    col_izq = int(10 + (4 - 10) * progreso)
    patron_triangulo[fila, col_izq] = 1
    col_der = int(10 + (16 - 10) * progreso)
    patron_triangulo[fila, col_der] = 1
patron_triangulo[4, 10] = 1

#PATRON ROMBO 20x20
patron_rombo = np.zeros((20, 20), dtype=int)
for i in range(8): 
    offset = i
    patron_rombo[3 + offset, 10 - offset] = 1
    patron_rombo[3 + offset, 10 + offset] = 1
    patron_rombo[17 - offset, 10 - offset] = 1
    patron_rombo[17 - offset, 10 + offset] = 1
patron_rombo[3, 10] = 1
patron_rombo[17, 10] = 1
patron_rombo[10, 3] = 1
patron_rombo[10, 17] = 1

#PATRON ESTRELLA 20x20
patron_estrella = np.zeros((20, 20), dtype=int)
patron_estrella[2, 10] = 1
patron_estrella[3, 10] = 1
patron_estrella[4, 11] = 1
patron_estrella[5, 11] = 1
patron_estrella[6, 12] = 1
patron_estrella[7, 12] = 1
patron_estrella[7, 13] = 1
patron_estrella[8, 14] = 1
patron_estrella[8, 15] = 1
patron_estrella[8, 16] = 1
patron_estrella[8, 17] = 1
patron_estrella[8, 18] = 1
patron_estrella[9, 17] = 1
patron_estrella[10, 16] = 1
patron_estrella[11, 15] = 1
patron_estrella[12, 15] = 1
patron_estrella[13, 14] = 1
patron_estrella[14, 15] = 1
patron_estrella[15, 15] = 1
patron_estrella[16, 16] = 1
patron_estrella[17, 16] = 1
patron_estrella[18, 16] = 1
patron_estrella[17, 14] = 1
patron_estrella[16, 13] = 1
patron_estrella[16, 12] = 1
patron_estrella[15, 11] = 1
patron_estrella[15, 10] = 1
patron_estrella[15, 9] = 1
patron_estrella[16, 8] = 1
patron_estrella[16, 7] = 1
patron_estrella[17, 6] = 1
patron_estrella[18, 4] = 1
patron_estrella[17, 4] = 1
patron_estrella[16, 4] = 1
patron_estrella[15, 5] = 1
patron_estrella[14, 5] = 1
patron_estrella[13, 6] = 1
patron_estrella[12, 5] = 1
patron_estrella[11, 5] = 1
patron_estrella[10, 4] = 1
patron_estrella[9, 3] = 1
patron_estrella[8, 2] = 1
patron_estrella[8, 3] = 1
patron_estrella[8, 4] = 1
patron_estrella[8, 5] = 1
patron_estrella[8, 6] = 1
patron_estrella[7, 7] = 1
patron_estrella[7, 8] = 1
patron_estrella[6, 8] = 1
patron_estrella[5, 9] = 1
patron_estrella[4, 9] = 1
patron_estrella[3, 10] = 1

#PATRON CRUZ 20x20
patron_cruz = np.zeros((20, 20), dtype=int)
patron_cruz[3:18, 10] = 1
patron_cruz[10, 3:18] = 1





def pintar_patron(patron):

    plt.figure(figsize=(4, 4))
    plt.imshow(patron, cmap='gray', vmin=0, vmax=1)
    plt.axis('off')
    plt.title(f"Patrón {patron.shape}")
    plt.show()
    
    

