


import streamlit as st
import pandas as pd

from PreprocesamientoLimpiezaDatos import PreprocesamientoLimpiezaDatos
from WebScrappingContinentes import WebScrappingContinentes
from AnalisisExploratorio import AnalisisExploratorio
from Clustering import Clustering



explicacion_columnas = [
                        'nombre de la liga a la que pertenece el equipo',
                        'duración de la partida (segundos)',
                        'nombre del equipo',
                        'resultado de la partida (1: ganar, 0: perder)',
                        'nº de muertes del equipo',
                        'nº de asistencias del equipo',
                        'nº de asesinatos a campeones enemigos del equipo',
                        '1 si el equipo ha conseguido el primer asesinato de la partida, 0 en caso contrario',
                        'nº de dragones asesinados por el equipo',
                        'nº de dragones infernales asesinados por el equipo',
                        'nº de dragones de montaña asesinados por el equipo',
                        'nº de dragones de nube asesinados por el equipo',
                        'nº de dragones de océano asesinados por el equipo',
                        'nº de dragones ancianos asesinados por el equipo',
                        '1 si se el equipo ha conseguido el primer dragón de la partida, 0 en caso contrario',
                        'nº de heraldos asesinados por el equipo',
                        '1 si se el equipo ha conseguido el primer heraldo de la partida, 0 en caso contrario',
                        'nº de barones Nashor asesinados por el equipo',
                        '1 si se el equipo ha conseguido el primer barón Nashor de la partida, 0 en caso contrario',
                        'nº de torretas destruidas por el equipo',
                        '1 si el equipo ha sido el primero en destruir una torreta, 0 en caso contrario',
                        '1 si el equipo ha sido el primero en destruir una torreta del carril central, 0 en caso contrario',
                        'nº de inhibidores destruidos por el equipo',
                        'cantidad total de daño hecho a campeones enemigos',
                        'nº de guardianes de visión colocados por el equipo',
                        'nº de guardianes de visión destruidos al equipo contrario',
                        'nº de súbditos asesinados por el equipo',
                        'nº de monstruos de la jungla asesinados por el equipo',
                        'nº de monstruos de la jungla del lado del mapa del equipo contrario asesinados por el equipo',
                        'cantidad de asesinatos a campeones enemigos en los 10 primeros minutos de partida',
                        'cantidad de oro en los 10 primeros minutos de partida'
                        ]
continentes_nombres = ['América del Norte', 'América del Sur', 'Europa', 'Asia', 'Oceania']
nombre_atributo_continente = 'continent'
nombre_atributo_lado_mapa = 'mapside'
enlace_web_lol_fandom_wiki = 'https://lol.fandom.com/wiki/'
enlace_web_wikipedia_ligas_lol = 'https://en.wikipedia.org/wiki/List_of_League_of_Legends_leagues_and_tournaments'
texto_seleccionar_dataset = 'Suba el conjunto de datos'
texto_indicar_caracteriscas_dataset = 'Indique el nombre de los atributos del conjunto de datos'
texto_seleccionar_seciones = 'Seleccione las secciones que quieres visualizar'
extension_fichero_conjunto_datos = {'csv'}
etiquetas_equipo = ['t1_', 't2_']

# Nombres de las secciones
nombre_seccion_atributos_utilizados = 'Atributos utilizados'
nombre_seccion_procesado_dataset = 'Procesado del conjunto de datos'
nombre_seccion_analisis_exploratorio_datos = 'Análisis exploratorio de datos'
nombre_seccion_analisis_continentes = 'Análisis exploratorio por continentes'
nombre_seccion_analisis_lado_mapa = 'Análisis exploratorio según el lado del mapa'
nombre_seccion_clustering = 'Clustering'




def mostrarTituloJustificado(importancia_titulo, texto):
    st.markdown('<div align="justify"> <h' + str(importancia_titulo) + '>' + texto + '</h + ' + str(importancia_titulo) + '> </div>', unsafe_allow_html = True)


def mostrarTextoJustificado(texto):
    st.markdown('<div align="justify"> <p>' + texto + '</p> </div>', unsafe_allow_html = True)


def mostrarTitulo(importancia_titulo, texto):
    st.markdown('<center> <h' + str(importancia_titulo) + '>' + texto + '</h + ' + str(importancia_titulo) + '> <center>', unsafe_allow_html = True)


@st.cache(show_spinner = False)
def columnasConEtiquetaDeEquipo(columnas, etiquetas_equipo):
    columnas_equipo = []

    # Columnas generales
    columnas_equipo.append(columnas[0])
    columnas_equipo.append(columnas[1])

    # Columnas dependientes del equipo
    for i in range(len(etiquetas_equipo)):
        for j in range(2, len(columnas)):
            columnas_equipo.append(etiquetas_equipo[i] + columnas[j])
    return columnas_equipo


def mostrarListaTexto(lista):
    lista_elementos = '<ul>'
    for elemento in lista:
        lista_elementos += '<li>' + elemento + '</li>'
    lista_elementos += '</ul>'
    
    st.markdown(lista_elementos, unsafe_allow_html = True)


def aumentarAnchuraBarraLateral():
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 600px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 600px;
            margin-left: -600px;
        }
        </style>
        """,
        unsafe_allow_html = True,
    )


@st.cache(show_spinner = False)
def explicacionesAtributos(columnas):
    explicacion_columnas_mostrar = []
    for columna, explicacion in zip(columnas, explicacion_columnas):
        explicacion_columnas_mostrar.append('<b>' + columna + '</b>: ' + explicacion)
    return explicacion_columnas_mostrar


@st.cache(show_spinner = False)
def explicacionesAtributosPrimeraLetraMayuscula():
    explicacion_columnas_mayus = []
    for columna in explicacion_columnas:
        explicacion_columnas_mayus.append(columna.capitalize())
    return explicacion_columnas_mayus


@st.cache(show_spinner = False)
def leerDatos(fichero_subido):
    return pd.read_csv(fichero_subido, sep = None)


@st.cache(allow_output_mutation = True, show_spinner = False)
def hacerPreprocesemientoAnalisisExploratorio(datos, columnas_equipo, columnas, atributos_categoricos):

    preprocesamiento_limpieza_datos = PreprocesamientoLimpiezaDatos(atributos_categoricos)

    # Acotar el dataset a las columnas seleccionadas
    datos_columnas_seleccionadas = datos[columnas_equipo]

    # Separar la partida en dos con la información de la partida de cada equipo
    datos_por_equipo = preprocesamiento_limpieza_datos.separarPartidaPorEquipos(datos_columnas_seleccionadas, columnas, nombre_atributo_lado_mapa)
    num_total_observaciones = datos_por_equipo.shape[0]

    analisis_exploratorio = AnalisisExploratorio(nombre_atributo_oro_10min = columnas[30], nombre_atributo_asesinatos_10min = columnas[29], nombre_atributo_dragones = columnas[8], nombre_atributo_barones = columnas[17], nombre_atributo_heraldos = columnas[15], nombre_atributo_lado_mapa = nombre_atributo_lado_mapa, nombre_atributo_resultado_partida = columnas[3], nombre_atributo_duracion_partida = columnas[1], nombre_atributo_continente = nombre_atributo_continente, nombre_atributo_equipo = columnas[2])

    # Distribución de los valores originales de los atributos
    figuras_distribucion_atributos_originales = analisis_exploratorio.figuraDistribucionAtributos(datos_por_equipo)

    # Crear un nuevo atributo mediante web scrapping: el continente
    datos_con_continentes = hacerWebScrapping(preprocesamiento_limpieza_datos, datos_por_equipo, nombre_atributo_equipo = columnas[2], nombre_atributo_liga = columnas[0])

    # La información geográfica y la duración de la partida no se incluye en la predicción de la victoria (no es una estrategia a seguir o es muy poco precisa)
    # nombre del equipo, nombre de la liga, duración de partida
    columnas_eliminar_predecir_victoria = [columnas[2], columnas[0], columnas[1]]
    datos_por_equipo.drop(columns = columnas_eliminar_predecir_victoria, axis = 1, inplace = True)
    predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria,  predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, num_observaciones_eliminadas_predecir_victoria = preprocesamiento_limpieza_datos.dividirDatosEnEntrenamientoTest(datos_por_equipo, 'result')

    figura_comparacion_observaciones_eliminadas = preprocesamiento_limpieza_datos.figuraComparacionObservaciones(num_total_observaciones, num_observaciones_eliminadas_predecir_victoria, datos_con_continentes.shape[0])

    # Análisis del porcentaje de victorias en función del lado del mapa
    figuras_analisis_lado_mapa = analisis_exploratorio.figurasAnalisisLadoMapa(datos_con_continentes)

    # Análisis exploratorio por continentes
    figuras_analisis_exploratorio_continentes = analisis_exploratorio.figurasAnalisisExploratorioContinentes(datos_con_continentes)

    # Codificar el atributo referente al continente (label encoding)
    datos_con_continentes_numerico = mapearContinentesNumerico(datos_con_continentes)

    # Escalar los datos con el atributo continente entre 0 y 1
    datos_con_continentes_numerico_escalado = preprocesamiento_limpieza_datos.escalarDatos(datos_con_continentes_numerico)

    return figuras_distribucion_atributos_originales, figuras_analisis_lado_mapa, figuras_analisis_exploratorio_continentes, figura_comparacion_observaciones_eliminadas, datos_con_continentes_numerico_escalado, datos_por_equipo, predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria, preprocesamiento_limpieza_datos


def hacerWebScrapping(preprocesamiento_limpieza_datos, datos_por_equipo, nombre_atributo_equipo, nombre_atributo_liga):
    datos_por_equipo = preprocesamiento_limpieza_datos.eliminarValoresPerdidos(datos_por_equipo)
    web_scrapping_continentes = WebScrappingContinentes(datos_por_equipo)
    datos_con_continentes, num_observaciones_no_encontradas_web_scrapping = web_scrapping_continentes.crearAtributoContinentes()
    # Se elimina la información geográfica porque se ha unificado en el atributo continente
    return datos_con_continentes.drop(columns = [nombre_atributo_equipo, nombre_atributo_liga], axis = 1).reset_index(drop = True)


def atributosCategoricas(columnas):
    return [columnas[3], columnas[7], columnas[14], columnas[16], columnas[18], columnas[20], columnas[21], nombre_atributo_continente, nombre_atributo_lado_mapa]


def mapearContinentesNumerico(datos):
    datos[nombre_atributo_continente] = pd.to_numeric(datos[nombre_atributo_continente].map({continentes_nombres[0]: 0, continentes_nombres[1]: 1, continentes_nombres[2]: 2,
                                                                    continentes_nombres[3]: 3, continentes_nombres[4]: 4}))
    return datos


def mostrarVariasFiguras(figuras):
    for figura in figuras:
        st.plotly_chart(figura)


def mostrarFigurasClustering(valor_k, figuras_clustering, figura_metodo_codo, precision_clasificacion_clusters, nombre_metrica):
    
    mostrarTituloJustificado(3, 'Determinar automáticamente el número de clusters utilizando como métrica ' + nombre_metrica + ' y como distancia la distancia de Gower')
    mostrarTitulo(4, 'Número de clusters elegido de forma automática: ' + str(valor_k))
    st.pyplot(figura_metodo_codo)

    mostrarTitulo(3, 'KMedoids con ' + str(valor_k) + ' clusters')

    mostrarTitulo(4, 'Cantidad de partidas por cluster')
    st.plotly_chart(figuras_clustering[0])   

    mostrarTitulo(4, 'Porcentaje de victorias por cluster')
    st.plotly_chart(figuras_clustering[1])   

    mostrarTituloJustificado(4, 'Para estimar los atributos que más difieren entre clusters se utiliza un modelo de bosque aleatorio para predecir el cluster al que pertenece cada partida. Para determinar la importancia de los atributos se utiliza la importancia por permutación. Precisión del modelo: ' + str(round(precision_clasificacion_clusters, 3)))
    mostrarTitulo(4, 'Atributos en los que más difieren los clusters')
    st.plotly_chart(figuras_clustering[2]) 

    mostrarTitulo(4, 'Comparación visual de los cinco atributos en los que más difieren los clusters')        
    mostrarVariasFiguras(figuras_clustering[3]) 

    mostrarTitulo(4, 'Distribución de valores de los cinco atributos en los que más difieren los clusters')
    mostrarVariasFiguras(figuras_clustering[4]) 

    mostrarTitulo(4, 'Gráfica que combina los dos atributos en los que más difieren los clusters')
    st.plotly_chart(figuras_clustering[5])


def mostrarFigurasAnalisisExploratorio(figuras_analisis_exploratorio, titulos_figuras):
    for i in range(len(figuras_analisis_exploratorio)):
        mostrarTitulo(3, titulos_figuras[i])
        st.plotly_chart(figuras_analisis_exploratorio[i])


@st.cache(allow_output_mutation = True, show_spinner = False)
def hacerClustering(datos, nombre_atributo_resultado, atributos_categoricos):
    clustering = Clustering(datos, nombre_atributo_resultado, atributos_categoricos)
    k_seleccionado_slhouette, k_seleccionado_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion = clustering.valoresSeleccionadosK()
    figuras_clustering_silhouette, precision_clasificacion_clusters_silhouette = clustering.getFigurasClustering(k_seleccionado_slhouette)
    figuras_clustering_distorsion, precision_clasificacion_clusters_distorsion = clustering.getFigurasClustering(k_seleccionado_distorsion)
    return k_seleccionado_slhouette, k_seleccionado_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion, figuras_clustering_silhouette, figuras_clustering_distorsion, precision_clasificacion_clusters_silhouette, precision_clasificacion_clusters_distorsion




def main():

    columnas_indicadas = False
    columnas = []


    if not columnas_indicadas:
        with st.expander(texto_indicar_caracteriscas_dataset):
            entrada_nombres_columnas = st.text_area('Indique los nombres de los atributos sin prefijos y separados por comas, en el orden en el que se indican abajo. Una vez inducidos, pulse Ctrl + Enter.',
                                                    'nombre_atributo_1, nombre_atributo_2, ...')
            columnas = [columna.strip() for columna in entrada_nombres_columnas.split(',')]


    if len(columnas) == 31:

        with st.expander(texto_seleccionar_dataset):
            # Subir el dataset
            fichero_subido = st.file_uploader('', type = extension_fichero_conjunto_datos, accept_multiple_files = False)


        if fichero_subido is not None:


            # Leer los datos
            datos = leerDatos(fichero_subido)

            columnas_equipo = columnasConEtiquetaDeEquipo(columnas, etiquetas_equipo)
            atributos_categoricos = atributosCategoricas(columnas)

            # Comprobar que el dataset subido contiene las columnas
            if set(columnas_equipo).issubset(datos.columns):

                columnas_indicadas = True

                with st.expander(nombre_seccion_atributos_utilizados):
                    mostrarTitulo(3, nombre_seccion_atributos_utilizados)
                    mostrarTituloJustificado(4, 'A continuación, se muestra una breve descripción de los atributos utilizados')
                    explicaciones_atributos_mostrar = explicacionesAtributos(columnas)
                    mostrarListaTexto(explicaciones_atributos_mostrar)


                # Preprocesamiento y análisis exploratorio
                figuras_distribucion_atributos_originales, figuras_analisis_lado_mapa, figuras_analisis_exploratorio_continentes, figura_comparacion_observaciones_eliminadas, datos_clustering,  datos_predecir_resultado_partida, predictores_entrenar_predecir_victoria, predictores_validacion_predecir_victoria, predictores_prueba_predecir_victoria, y_entrenar_predecir_victoria, y_validacion_predecir_victoria, y_prueba_predecir_victoria, predictores_entrenamiento_validacion_predecir_victoria, y_entrenamiento_validacion_predecir_victoria, indices_entrenamiento_validacion_predecir_victoria, predictores_entrenamiento_sin_escalar_predecir_victoria, y_entrenar_sin_escalar_predecir_victoria, preprocesamiento_limpieza_datos = hacerPreprocesemientoAnalisisExploratorio(datos, columnas_equipo, columnas, atributos_categoricos)

                # Clustering 
                k_seleccionado_slhouette, k_seleccionado_distorsion, figura_metodo_codo_silhouette, figura_metodo_codo_distorsion, figuras_clustering_silhouette, figuras_clustering_distorsion, precision_clasificacion_clusters_silhouette, precision_clasificacion_clusters_distorsion = hacerClustering(datos_clustering, nombre_atributo_resultado = columnas[3], atributos_categoricos = atributos_categoricos)
                

                # Selector de secciones
                with st.sidebar:
                    secciones_seleccionadas = st.multiselect(
                                            texto_seleccionar_seciones,
                                            [nombre_seccion_analisis_exploratorio_datos, nombre_seccion_analisis_lado_mapa, nombre_seccion_analisis_continentes, nombre_seccion_clustering])
                aumentarAnchuraBarraLateral()


                # Mostrar la información en función de las secciones seleccionadas

                # Procesamiento del conjunto de datos
                with st.expander(nombre_seccion_procesado_dataset):
                    mostrarTitulo(2, nombre_seccion_procesado_dataset)
                    mostrarTitulo(3, 'Observaciones eliminadas respecto al total')
                    mostrarTituloJustificado(4, 'Nótese que la información de la partida se ha separado por equipos, por lo que si se quiere saber la información por partida en general habría que dividir entre dos.')
                    st.plotly_chart(figura_comparacion_observaciones_eliminadas)

                # Mostrar el análisis exploratorio de datos
                if nombre_seccion_analisis_exploratorio_datos in secciones_seleccionadas:
                    with st.expander(nombre_seccion_analisis_exploratorio_datos):
                        mostrarTitulo(2, nombre_seccion_analisis_exploratorio_datos)
                        mostrarTitulo(3, 'Distribución de los valores de los atributos originales')
                        mostrarVariasFiguras(figuras_distribucion_atributos_originales)

                # Mostrar el análisis por continentes
                if nombre_seccion_analisis_continentes in secciones_seleccionadas:
                    with st.expander(nombre_seccion_analisis_continentes):
                        mostrarTitulo(2, nombre_seccion_analisis_continentes)
                        mostrarTituloJustificado(3, 'La información de los continentes se ha encontrado mediante web scrapping en las webs: ')
                        columna_izquierda, columna_central, columna_derecha = st.columns(3)
                        st.markdown('##', unsafe_allow_html = True)
                        with columna_central:
                            st.write('- [Lol Fandom Wiki](' + enlace_web_lol_fandom_wiki + ')')
                            st.write('- [Torneos de LoL (Wikipedia)](' + enlace_web_wikipedia_ligas_lol + ')')
                        titulos_figuras_analisis_continentes = ['Número de partidas de cada continente', 'Cantidad de oro ganado en los 10 primeros minutos en función del continente', 'Cantidad de asesinatos a campeones enemigos en los 10 primeros minutos en función del continente', 'Cantidad de monstruos grandes (dragones, barones Nashor y heraldos) en función del continente']
                        mostrarFigurasAnalisisExploratorio(figuras_analisis_exploratorio_continentes, titulos_figuras_analisis_continentes)

                # Mostrar el ánalisis según el lado del mapa
                if nombre_seccion_analisis_lado_mapa in secciones_seleccionadas:
                    with st.expander(nombre_seccion_analisis_lado_mapa):
                        mostrarTitulo(2, nombre_seccion_analisis_lado_mapa)
                        titulo_figuras_analisis_lado_mapa = ['Porcentaje de victorias general en función del lado del mapa', 'Porcentaje de victorias por cuartil de duración (lado azul)', 'Porcentaje de victorias por cuartil de duración (lado rojo)']
                        mostrarFigurasAnalisisExploratorio(figuras_analisis_lado_mapa, titulo_figuras_analisis_lado_mapa)

                # Mostrar el clustering 
                if nombre_seccion_clustering in secciones_seleccionadas:
                    with st.expander(nombre_seccion_clustering):
                        mostrarTitulo(2, nombre_seccion_clustering)
                        mostrarTitulo(3, 'Método para realizar el clustering: KMedoides')
                        mostrarFigurasClustering(k_seleccionado_slhouette, figuras_clustering_silhouette, figura_metodo_codo_silhouette, precision_clasificacion_clusters_silhouette, 'Silhouette')
                        mostrarFigurasClustering(k_seleccionado_distorsion, figuras_clustering_distorsion, figura_metodo_codo_distorsion, precision_clasificacion_clusters_distorsion, 'la distorsión')

        

    else:
        mostrarTextoJustificado('No ha indicado las columnas correctamente. Por favor, indica el nombre de las columnas que se describen en la parte inferior.')
        mostrarTextoJustificado('El conjunto de datos debe estar en formato CSV y almacenar la información de la partida en cada fila. Debe contener los siguientes atributos en el orden en el que se indican, por cada uno de los dos equipos de esa partida, estando en primer lugar todos los atributos referentes al equipo azul y después los del equipo rojo. Para indicar que los atributos se refieren al equipo azul, debe escribirse el prefijo t1_ delante del atributo, siendo t2_ el prefijo para el equipo rojo.')
        mostrarListaTexto(explicacionesAtributosPrimeraLetraMayuscula())
        



if __name__ == '__main__':
    main()
