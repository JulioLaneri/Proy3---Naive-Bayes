#
# Universidad Nacional de Itapua
# Inteligencia Artificial - 7mo Semestre
#
# Proyecto#3: Clasificador de Bayes
#
# Alumno: _____________
#

from os import listdir
from os.path import isfile,join
import re
from math import log

##
# Función auxiliar para extraer palabras de un archivo
#
def extraer_palabras(archivo):
    """
    Lee un archivo y extrae todas las palabras en minúsculas
    """
    try:
        with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
            contenido = f.read()
            # Extraer solo palabras (letras y números), convertir a minúsculas
            palabras = re.findall(r'\b[a-z0-9]+\b', contenido.lower())
            return palabras
    except Exception as e:
        return []

##
# Función auxiliar para contar palabras
#
def contar_palabras(archivos):
    """
    Cuenta la frecuencia de cada palabra en una lista de archivos
    Retorna un diccionario {palabra: frecuencia}
    """
    conteo = {}
    for archivo in archivos:
        palabras = extraer_palabras(archivo)
        for palabra in palabras:
            conteo[palabra] = conteo.get(palabra, 0) + 1
    return conteo

##
#  Esto entrena el modelo
#
#  modelo: el modelo actual que debes modificar o None si ninguno existe todavia
#  archivos: una lista de nombres de archivos
#  clase: la clase que corresponde a este archivo (Ej: 'comp.os.ms-windows.misc')
#  K: el factor de Laplace Smoothing
#
def entrenar(modelo, archivos, clase, K):
    """
    Entrena el modelo de Naive Bayes con los archivos dados
    
    El modelo es un diccionario con:
    - 'clases': dict con info de cada clase
        - 'conteo_palabras': dict {palabra: frecuencia}
        - 'total_palabras': total de palabras en la clase
        - 'num_documentos': número de documentos de esta clase
    - 'vocabulario': set con todas las palabras únicas (diccionario global)
    - 'total_documentos': total de documentos entrenados
    """
    
    # Inicializar modelo si no existe
    if modelo is None:
        modelo = {
            'clases': {},
            'vocabulario': set(),
            'total_documentos': 0
        }
    
    # Inicializar clase si no existe
    if clase not in modelo['clases']:
        modelo['clases'][clase] = {
            'conteo_palabras': {},
            'total_palabras': 0,
            'num_documentos': 0
        }
    
    # Contar palabras en los archivos de esta clase
    conteo_clase = contar_palabras(archivos)
    
    # Actualizar el modelo
    for palabra, freq in conteo_clase.items():
        # Agregar palabra al vocabulario global
        modelo['vocabulario'].add(palabra)
        
        # Actualizar conteo de palabras de la clase
        if palabra in modelo['clases'][clase]['conteo_palabras']:
            modelo['clases'][clase]['conteo_palabras'][palabra] += freq
        else:
            modelo['clases'][clase]['conteo_palabras'][palabra] = freq
        
        # Actualizar total de palabras de la clase
        modelo['clases'][clase]['total_palabras'] += freq
    
    # Actualizar número de documentos
    modelo['clases'][clase]['num_documentos'] += len(archivos)
    modelo['total_documentos'] += len(archivos)
    
    return modelo


##
# Esto hace la clasificacion usando el modelo
#
# modelo: el modelo a utilizar para clasificar
# archivo: el archivo que vas a clasificar
# K: el valor K
#
#
def clasificar(modelo, archivo, K):
    """
    Clasifica un archivo usando el modelo de Naive Bayes
    
    Usa la fórmula de Bayes:
    P(clase|documento) ∝ P(clase) * ∏ P(palabra|clase)
    
    Con suavizado de Laplace:
    P(palabra|clase) = (conteo(palabra, clase) + K) / (total_palabras_clase + K * |vocabulario|)
    """
    
    if modelo is None or not modelo['clases']:
        return ''
    
    # Extraer palabras del archivo
    palabras = extraer_palabras(archivo)
    
    if not palabras:
        # Si no hay palabras, retornar la clase más probable (la que tiene más documentos)
        return max(modelo['clases'].items(), key=lambda x: x[1]['num_documentos'])[0]
    
    # Calcular el tamaño del vocabulario
    V = len(modelo['vocabulario'])
    
    # Calcular probabilidades para cada clase
    mejor_clase = None
    mejor_log_prob = float('-inf')
    
    for clase, info_clase in modelo['clases'].items():
        # P(clase) = num_documentos_clase / total_documentos
        prob_clase = info_clase['num_documentos'] / modelo['total_documentos']
        
        # Usar logaritmos para evitar underflow
        log_prob = log(prob_clase)
        
        # Calcular P(documento|clase) = ∏ P(palabra|clase)
        # En logaritmos: log(P(documento|clase)) = Σ log(P(palabra|clase))
        
        total_palabras_clase = info_clase['total_palabras']
        conteo_palabras_clase = info_clase['conteo_palabras']
        
        for palabra in palabras:
            # Suavizado de Laplace
            # P(palabra|clase) = (conteo(palabra, clase) + K) / (total_palabras_clase + K * V)
            conteo_palabra = conteo_palabras_clase.get(palabra, 0)
            
            # Asegurar que K sea al menos 1 para evitar división por cero
            K_usado = max(K, 1)
            
            prob_palabra_dado_clase = (conteo_palabra + K_usado) / (total_palabras_clase + K_usado * V)
            
            # Solo agregar si la probabilidad es válida (mayor que 0)
            if prob_palabra_dado_clase > 0:
                log_prob += log(prob_palabra_dado_clase)
        
        # Actualizar mejor clase si esta tiene mayor probabilidad
        if log_prob > mejor_log_prob:
            mejor_log_prob = log_prob
            mejor_clase = clase
    
    return mejor_clase if mejor_clase else ''
