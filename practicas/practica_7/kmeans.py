#!/usr/bin/python3
#Práctica K-MEANS
# a)Buscar el DATASET del TITANIC, elegir 3 características y correr k-means( con una adecuada selección de parametros)
#   y que haga 2 clusters (Vivo o Muerto) y compare con el original
# b) Hacer lo mismo con IRIS

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import argparse
import sys


#FUNCIÓN PARA LEER EL DATASET
def leer_csv(nombre_dataset):
    try:
        datasetDF = pd.read_csv(nombre_dataset)
    except FileNotFoundError:
        print("NO SE ENCONTRO EL ARCHIVO CSV :(")
        sys.exit()
        
    return datasetDF

#FUNCIÓN PARA LIMPIAR DATOS

def limpiar_datos(dataset):
    datasetDF = dataset.copy()
    #SE USAN NÚMEROS PARA EL SEXO DE LOS PASAJEROS
    if 'sex' in datasetDF.columns:
        datasetDF['sex'] = datasetDF['sex'].map({'female': 1, 'male': 0})
    #LAS EDADES NO REGISTRADAS SE ASIGNAN CON LA MEDIA DE EDADES
    if 'age' in datasetDF.columns:
        edad_media = datasetDF['age'].median()
        datasetDF['age'] = datasetDF['age'].fillna(edad_media)
    if 'Species' in datasetDF.columns:
        datasetDF['Species'] = datasetDF['Species'].map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})
    
    return datasetDF


#FUNCIÓN PARA OBTENER LA MÉTRICA EUCLIDIANA
def distancia_euclidiana(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

#ALGORITMO KMEANS
def kmeans(titanic_df, k, epocas):
    #USAMOS ARRAY Y OBTENEMOS DIMENSIONES
    titanic_array = np.array(titanic_df)
    num_datos, num_caracteristicas = titanic_array.shape
    #CENTROIDES ALEATORIOS
    indices = np.random.choice(num_datos, size=k, replace=False)
    centroides = titanic_array[indices]
    cluster_asignados = [0]*num_datos
    #BUSQUEDA DE CENTROIDES EN N EPOCAS
    for iteration in range(epocas):
        #CLUSTER MÁS CERCANO
        
        for i in range(num_datos):
            #SE CALCULA LA DISTANCIA A CADA K CENTROIDE
            distancias = [distancia_euclidiana(centroides[j], titanic_array[i]) for j in range(k)]
            distancia_minima = np.argmin(distancias)
            cluster_asignados[i] = distancia_minima
        #RECALCULO DE CENTROIDES
        centroides_nuevos = []
        for cluster_index in range(k):
            puntos_del_cluster = titanic_array[np.array(cluster_asignados) == cluster_index]
            if len(puntos_del_cluster) > 0:
                cluster_mean = np.mean(puntos_del_cluster, axis=0)
                centroides_nuevos.append(cluster_mean)
            else:#EL CLUSTER ESTA VACION, ENTONCES MANTEN EL CENTROIDE ANTERIOR
                centroides_nuevos.append(centroides[cluster_index])
            
        centroides_nuevos = np.array(centroides_nuevos)
        
        if np.allclose(centroides, centroides_nuevos, atol=1e-9): 
            #print(f"convergencia en la epoca={iteration+1}")
            centroides = centroides_nuevos
            break
        #SI NO HAY CONVERGENCIA ENTONCES ACTUALIZAMOS CENTROIDES PARA LA PRÓXIMA ITERACIÓN
        centroides = centroides_nuevos
        #ASIGNACION DE CLUSTERS
        clusters = [[] for _ in range(k)]
        for i in range(num_datos):
            clusters[cluster_asignados[i]].append(titanic_array[i].tolist())
            
    return centroides.tolist(), clusters, cluster_asignados
    
#FUNCIÓN PARA GRAFICAR MATRIZ
def graficar_matriz_confusion(Y, Y_pred):
    cm = confusion_matrix(Y, Y_pred)
    cm_df = pd.DataFrame(cm,
                        index=[f'Real: Murió (0)', f'Real: Sobrevivió (1)'],
                        columns=[f'K-Means: Cluster 0', f'K-Means: Cluster 1'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Comparación: Realidad vs K-Means')
    plt.ylabel('Etiqueta Real')
    plt.xlabel('Clusters de K-Means')
    plt.show()
    
    return cm
    


#FUNCIÓN PARA GRAFICAR CLUSTERS
def graficar_clusters_3d_original(datos_originales_df, Y_pred, centroides_escalados, scaler_obj, feature_names, cm):
    #SE DES-ESCALAN LOS CENTROIDES
    centroides_originales = scaler_obj.inverse_transform(centroides_escalados)
    #CONEVRTIR DF A ARRAY
    datos_originales = datos_originales_df.values
    Y_pred = np.array(Y_pred)
    #GŔAFICO
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue']
    #OBTENCIÓN DE ETIQUETAS PARA LOS CLUSTERS MEDIANTE LA MATRIZ DE CONFUSIÓN
    if cm[0, 0] > cm[1, 0]:
        cluster_labels = {0: 'Muertos (Cluster 0)', 1: 'Vivos (Cluster 1)'}
    else:
        cluster_labels = {0: 'Vivos (Cluster 0)', 1: 'Muertos (Cluster 1)'}
    #CLUSTER 0
    cluster_0 = datos_originales[Y_pred == 0]
    ax.scatter(cluster_0[:, 0], cluster_0[:, 1], cluster_0[:, 2], 
               color=colors[0], label=cluster_labels[0], alpha=0.3)
    #CLUSTER 1
    cluster_1 = datos_originales[Y_pred == 1]
    ax.scatter(cluster_1[:, 0], cluster_1[:, 1], cluster_1[:, 2], 
               color=colors[1], label=cluster_labels[1], alpha=0.3)
    #CENTROIDES
    ax.scatter(centroides_originales[:, 0], centroides_originales[:, 1], centroides_originales[:, 2],
               color='black', marker='x', s=200, label='Centroides', linewidth=3, depthshade=False)
    #NOMBRES DE LOS EJES
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    #AJUSTE DE EJES
    if feature_names[0] == 'Clase':
        ax.set_xticks([1, 2, 3]) 
        ax.set_xticklabels(['Primera', 'Segunda', 'Tercera']) 
        
    if feature_names[2] == 'Sexo':
        ax.set_zticks([0, 1]) 
        ax.set_zticklabels(['Hombre', 'Mujer']) 
        
    ax.set_title('K-Means ')
    ax.legend()
    plt.show()
    
    
#FUNCIÓN PARA GRAFICAR MATRIZ (IRIS, k=3)
def graficar_matriz_confusion_iris(Y, Y_pred):
    cm = confusion_matrix(Y, Y_pred)
    cm_df = pd.DataFrame(cm,
                         index=[f'Real: Iris-setosa', f'Real: Iris-versicolor', f'Real: Iris-virginica'],
                        columns=[f'K-Means: Cluster 0', f'K-Means: Cluster 1', f'K-Means: Cluster 2'])
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues')
    plt.title('Comparación: Realidad vs K-Means (Iris)')
    plt.ylabel('Especie Real')
    plt.xlabel('Clusters de K-Means')
    plt.show()
    return cm

#FUNCIÓN PARA GRAFICAR CLUSTERS (IRIS, k=3)
def graficar_clusters_3d_original_iris(datos_originales_df, Y_pred, centroides_escalados, scaler_obj, feature_names):
    #SE DES-ESCALAN LOS CENTROIDES
    centroides_originales = scaler_obj.inverse_transform(centroides_escalados)
    #CONEVRTIR DF A ARRAY
    datos_originales = datos_originales_df.values
    Y_pred = np.array(Y_pred)
    k = 3 # k=3 para Iris
    
    #GŔAFICO
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    colors = ['red', 'blue', 'green'] # 3 colores
    
    # Bucle para graficar los k clústeres
    for cluster_index in range(k):
        cluster_data = datos_originales[Y_pred == cluster_index]
        ax.scatter(cluster_data[:, 0], cluster_data[:, 1], cluster_data[:, 2], 
                   color=colors[cluster_index], label=f'Cluster {cluster_index}', alpha=0.3)

    #CENTROIDES
    ax.scatter(centroides_originales[:, 0], centroides_originales[:, 1], centroides_originales[:, 2],
               color='black', marker='x', s=200, label='Centroides', linewidth=3, depthshade=False)
    #NOMBRES DE LOS EJES
    ax.set_xlabel(feature_names[0])
    ax.set_ylabel(feature_names[1])
    ax.set_zlabel(feature_names[2])
    
    ax.set_title('K-Means (Iris)')
    ax.legend()
    plt.show()

def main():
    #ARGUMENTOS CLI
    parser = argparse.ArgumentParser(description="Algoritmo de aprendizaje no supervizado K-Means")
    parser.add_argument("-t", "--titanic", action="store_true", help="Clusterizar dataset del Titanic")
    parser.add_argument("-i", "--iris", action="store_true", help="Clusterizar dataset Iris")
    args = parser.parse_args()



    #K-MEANS PARA TITANIC
    if args.titanic:
        #LECTURA CSV Y LIMPIEZA DE DATOS
        titanicDF_LIMPIO = limpiar_datos(leer_csv('titanic.csv'))
        #CARACTERISTICAS USADAS Y OBJETIVO CON EL CUAL SE HARA LA COMPARACIÓN
        caracteristicas = ['pclass', 'age', 'sex']
        objetivo = 'survived'
        #DATOS FINALES ALINEADOS PORQUE SE BORRARON LOS NaN
        titanicDF = titanicDF_LIMPIO[caracteristicas + [objetivo]].dropna().reset_index(drop=True)
        #SEPARAMOS CARACTERISTICAS QUE SE BUSCAN AGRUPAR Y LA ETIQUETA QUE QUEREMOS COMPARAR
        Y = titanicDF[objetivo]
        X = titanicDF[caracteristicas]
        #ESCALANDO DATOS
        escala = StandardScaler()
        X_escalado = escala.fit_transform(X)
        #EJECUCIÓN DE KMEANS
        centroides, clusters, Y_pred = kmeans(X_escalado, 2, 100)
        #GRAFICA DE CONFUSIÓN
        print(Y)
        cm=graficar_matriz_confusion(Y, Y_pred)
        #GRAFICAR LOS CLUSTERS OBTENIDOS
        graficar_clusters_3d_original(X, Y_pred, centroides, escala, ['Clase', 'Edad', 'Sexo'], cm)

    #K-MEANS PARA IRIS
    elif args.iris:
        #LECTURA CSV Y LIMPIEZA DE DATOS
        irisDF = limpiar_datos(leer_csv('Iris.csv'))
        #CARACTERISTICAS USADAS Y OBJETIVO CON EL CUAL SE HARA LA COMPARACIÓN
        caracteristicas = ['SepalLengthCm','SepalWidthCm','PetalLengthCm']
        objetivo = 'Species'
        #SEPARAMOS CARACTERISTICAS QUE SE BUSCAN AGRUPAR Y LA ETIQUETA QUE QUEREMOS COMPARAR
        Y = irisDF[objetivo]
        X = irisDF[caracteristicas]
        #ESCALANDO DATOS
        escala = StandardScaler()
        X_escalado = escala.fit_transform(X)
        #EJECUCIÓN DE KMEANS
        centroides, clusters, Y_pred = kmeans(X_escalado, 3, 100)
        #MATRIZ DE CONFUSION
        graficar_matriz_confusion_iris(Y, Y_pred) 
        #GRAFICAR CLUSTERS
        graficar_clusters_3d_original_iris(X, Y_pred, centroides, escala, ['Longitud del Sépalo','Ancho del Sépalo','Longitud del Pétalo'])
    
    if not(args.titanic or args.iris):
        print("No se recibió una bandera válida :(")
        print("Use './kmeans -h' para ver las opciones disponibles")

if __name__ == '__main__':
    main()
    

