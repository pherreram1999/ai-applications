import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import sys

#LECTURA CSV
def leer_csv(nombre, caracteristicas, etiquetas):
    try:
        df = pd.read_csv(nombre)
        X_df = df[caracteristicas]
        Y_df = df[etiquetas]
        X_csv = X_df.values
        Y_csv = Y_df.values

    except FileNotFoundError:
        print(".:Archivo no encontrado:.")
        sys.exit()
    except KeyError as e:
        print(f".:Columna {e} no encontrada:.")
        sys.exit()

    return X_csv[:,0], X_csv[:,1], X_csv[:,2], Y_csv[:,0]

#FUNCIÓN DE NORMALIZACIÓN
def normalizar(X1,X2,X3,Y, extremos):
    X1_norm = (X1 - extremos['X1_min']) / (extremos['X1_max'] - extremos['X1_min'])
    X2_norm = (X2 - extremos['X2_min']) / (extremos['X2_max'] - extremos['X2_min'])
    X3_norm = (X3 - extremos['X3_min']) / (extremos['X3_max'] - extremos['X3_min'])
    Y_norm =  (Y  - extremos['Y_min']) / (extremos['Y_max']  - extremos['Y_min'])
    
    return X1_norm,X2_norm,X3_norm,Y_norm

#FUNCIÓN DE DES-NORMALIZADO
def desnormalizar(X1,X2,X3,Y,b0n,b1n,b2n,b3n, extremos):
    b0 = (extremos['Y_max'] - extremos['Y_min'])*(
        b0n - b1n*(extremos['X1_min']/(extremos['X1_max']-extremos['X1_min'])) 
            - b2n*(extremos['X2_min']/(extremos['X2_max']-extremos['X2_min'])) 
            - b3n*(extremos['X3_min']/(extremos['X3_max']-extremos['X3_min']))
    ) + extremos['Y_min']
    b1 = b1n*((extremos['Y_max'] - extremos['Y_min'])/(extremos['X1_max'] - extremos['X1_min']))
    b2 = b2n*((extremos['Y_max'] - extremos['Y_min'])/(extremos['X2_max'] - extremos['X2_min']))
    b3 = b3n*((extremos['Y_max'] - extremos['Y_min'])/(extremos['X3_max'] - extremos['X3_min']))

    return b0,b1,b2,b3
    

#FUNCIÓN DE REGRESIÓN DE RIDGE
def regresion_ridge(lr, T, epsilon, epocas_max, X1, X2, X3, Y):
    #INICIALIZACIÓN DE REGRESORES
    b0 = 0
    b1 = 0
    b2 = 0
    b3 = 0
    #INICIALIZACIÓN DE VARIABLES
    m = len(Y)
    J = 0
    Jant = 1000
    epocas = 0
    ECM = []
    #ALGORITMO DE RIDGE
    while(epocas<epocas_max and abs(J-Jant)>epsilon):
        #MODELO
        Ym = b0 + b1*X1 + b2*X2 + b3*X3
        #CÁLCULO DE LA FUNCIÓN DE COSTOS
        Jant = J
        residuales = Ym - Y
        costo_mse = (1/(2*m))*np.sum(residuales**2)
        costo_penalizacion = T*(b1**2 + b2**2 + b3**2)
        J = costo_mse + costo_penalizacion
        ECM.append(J)
        #DESCENSO DE GRADIENTE
        grad_b0 = (1/m)*np.sum(residuales)
        grad_b1_mse = (1/m)*np.dot(residuales,X1)
        grad_b2_mse = (1/m)*np.dot(residuales,X2)
        grad_b3_mse = (1/m)*np.dot(residuales,X3)
        grad_b1_penalizacion = 2*T*b1
        grad_b2_penalizacion = 2*T*b2
        grad_b3_penalizacion = 2*T*b3
        grad_b1 = grad_b1_mse + grad_b1_penalizacion
        grad_b2 = grad_b2_mse + grad_b2_penalizacion
        grad_b3 = grad_b3_mse + grad_b3_penalizacion
        #ACTUALIZACIÓN DE REGRESORES
        b0 = b0 - lr*grad_b0
        b1 = b1 - lr*grad_b1
        b2 = b2 - lr*grad_b2
        b3 = b3 - lr*grad_b3
        #EPOCAS++
        epocas = epocas + 1
        #PRUEBAS
        #print(f"Epoca número: {epocas}")
        #print(f"B0={b0} B1={b1} B2={b2} B3={b3}")
    #print(f"J={J}")
    return b0,b1,b2,b3,ECM


#FUNCIÓN PARA GRAFICAR MODELO Y CONVERGENCIA
def graficar_convergencia(X, Y, b0, b1, historial_costos):
    ##GRÁFICO DE FUNCIÓN DE CONVERGENCIA
    plt.figure(figsize=(12,5))
    plt.plot(historial_costos, label="Función de Costos (J)")
    plt.title("Convergencia del Descenso de Gradiente")
    plt.xlabel("Épocas")
    plt.ylabel("Costo (J)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def graficar_plano(X1, X2, X3, Y, b0, b1, b2, b3, var1_name, var2_name):
    #DATAFRAME
    df = pd.DataFrame({'Reactor Temperature': X1, 'Ratio of H2 to n-Heptane': X2, 'Contact Time': X3, 'Conversion of n-Heptane to Acetylene': Y})
    
    #CARACTERISTICAS
    all_vars = {'Reactor Temperature', 'Ratio of H2 to n-Heptane', 'Contact Time'}
    dynamic_vars = {var1_name, var2_name}
    
    #VARIABLE FIJA
    fixed_var_name = list(all_vars - dynamic_vars)[0]
    
    #REGRESORES
    betas = {'Reactor Temperature': b1, 'Ratio of H2 to n-Heptane': b2, 'Contact Time': b3}
    
    #SE HACE CONSTANTE Xn 
    var1_data = df[var1_name]
    var2_data = df[var2_name]
    fixed_var_mean = df[fixed_var_name].mean()
    
    #GRID
    var1_grid = np.linspace(var1_data.min(), var1_data.max(), len(df))
    var2_grid = np.linspace(var2_data.min(), var2_data.max(), len(df))
    var1_malla, var2_malla = np.meshgrid(var1_grid, var2_grid)

    #PLANO Ym
    y_plano = (
        betas[var1_name] * var1_malla +
        betas[var2_name] * var2_malla +
        betas[fixed_var_name] * fixed_var_mean +
        b0
    )

    fig_combinada = go.Figure()

    fig_combinada.add_trace(go.Scatter3d(
        x=var1_data, 
        y=var2_data, 
        z=df['Conversion of n-Heptane to Acetylene'],  
        mode='markers',
        marker=dict(size=4, color='red', opacity=0.7),
        name='Datos reales'
    ))

    fig_combinada.add_trace(go.Surface(
        x=var1_malla,   
        y=var2_malla,   
        z=y_plano,     
        name='Plano de Ym',
        opacity=0.7,
        colorscale='cividis'
    ))

    fig_combinada.update_layout(
        title=f"({var1_name.upper()} Y {var2_name.upper()}) VS MODELO",
        scene=dict(
            xaxis_title=var1_name.upper(),
            yaxis_title=var2_name.upper(),
            zaxis_title='Conversion of n-Heptane to Acetylene (%)',
        )
    )

    fig_combinada.show()
    
    
def graficar_caracteristicas(X1, X2, X3):

    df_features = pd.DataFrame({'x1': X1, 'x2': X2, 'x3': X3})
    
    fig = px.scatter_3d(
        df_features,
        x='x1',
        y='x2',
        z='x3',
        title="Distribución de características",
        opacity=0.7
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='Reactor Temperature'),
            yaxis=dict(title='Ratio of H2 to n-Heptane'),
            zaxis=dict(title='Contact Time'),
        )
    )
    
    fig.show()
    

def main():     
    #ARGUMENTOS CLI
    parser = argparse.ArgumentParser(description="Regresion de Ridge :)")
    parser.add_argument("-g", "--graficas", action="store_true", help="Desplegar Gráficos")
    parser.add_argument("-p", "--predecir", nargs=3, type=float, metavar=('X1', 'X2', 'X3'), help="Fase de predicción")
    parser.add_argument("-c", "--clasificar", nargs=1, type=float, metavar=('Y'), help="Fase de clasificación")
    args = parser.parse_args()
    
    #PARÁMETROS
    lr = 0.3
    T = 1e-6
    epsilon = 1e-12
    epocas_max = 100000
    
    #CARACTERISTICAS
    x1 = 'Reactor Temperature'
    x2 = 'Ratio of H2 to n-Heptane'
    x3 = 'Contact Time'
    
    #ETIQUETAS
    y = 'Conversion of n-Heptane to Acetylene'
    
    #CARACTERÍSTICAS Y ETIQUETA
    caracteristicas = [x1, x2, x3]
    etiquetas = [y]
    #LECTURA CSV
    X1,X2,X3,Y = leer_csv('dataset_acetileno.csv', caracteristicas, etiquetas)
    
    #ALMACENAMIENTO DE EXTREMOS
    extremos = {
            "X1_min": X1.min(),
            "X1_max": X1.max(),
            "X2_min": X2.min(),
            "X2_max": X2.max(),
            "X3_min": X3.min(),
            "X3_max": X3.max(),
            "Y_min" : Y.min(),
            "Y_max" : Y.max()
        }
    
    #NORMALIZACIÓN
    X1_norm,X2_norm,X3_norm,Y_norm = normalizar(X1,X2,X3,Y, extremos)
    
    #REGRESIÓN DE RIDGE
    b0n,b1n,b2n,b3n,historial_costos = regresion_ridge(lr, T, epsilon, epocas_max, X1_norm, X2_norm, X3_norm, Y_norm)
    
    #DES-NORMALIZACIÓN
    b0,b1,b2,b3 = desnormalizar(X1,X2,X3,Y,b0n,b1n,b2n,b3n, extremos)
    
    #GRÁFICA DE CONVERGENCIA DE J
    #graficar_convergencia(X1, Y, b0, b1, historial_costos)
    
    #MODELO
    print("-"*50)
    print(f"Modelo: Y = {b0:.4f} + ({b1:.4f})*X1 + ({b2:.4f})*X2 + ({b3:.4f})*X3")
    print("-"*50)
    
    #DECISIÓN SEGUN ARGUMENTOS CLI
    if args.graficas:
        #GRAFICAR PLANOS
        graficar_plano(X1,X2,X3,Y,b0,b1,b2,b3,var1_name=x1,var2_name=x2)
        graficar_plano(X1,X2,X3,Y,b0,b1,b2,b3,var1_name=x1,var2_name=x3)
        graficar_plano(X1,X2,X3,Y,b0,b1,b2,b3,var1_name=x2,var2_name=x3)
        graficar_caracteristicas(X1,X2,X3)
    
    if args.predecir:
        val_x1, val_x2, val_x3 = args.predecir
        prediccion = b0 + b1*val_x1 + b2*val_x2 + b3*val_x3
        print(f"Entradas -> Temp: {val_x1}, Ratio: {val_x2}, Time: {val_x3}")
        print(f"Prediccion (Ym): {prediccion:.4f}")
        
    if args.clasificar:
        print("Ya es toda,gue")
    
    if not(args.predecir or args.clasificar or args.graficas):
        print("No se recibio bandera de acción :(")
        print("Use 'python3 ridge.py -h' para ver las opciones disponibles")


if __name__ == "__main__":
    main()
    
