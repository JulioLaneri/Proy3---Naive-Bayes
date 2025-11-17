#
# Universidad Nacional de Itapua
# Inteligencia Artificial - 7mo Semestre
#
# Proyecto#3: Clasificador de Bayes
#
# Alumno: Julio Daniel Benitez Laneri
#


from os import listdir
from os.path import isfile,join
import re
from math import log


def extraer_palabras(archivo):
    try:
        with open(archivo, 'r', encoding='utf-8', errors='ignore') as f:
            contenido = f.read()
            palabras = re.findall(r'\b[a-z0-9]+\b', contenido.lower())
            return palabras
    except Exception as e:
        return []


def contar_palabras(archivos):
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
    
    if modelo is None:
        modelo = {
            'clases': {},
            'vocabulario': set(),  
            'total_documentos': 0
        }
    
    if clase not in modelo['clases']:
        modelo['clases'][clase] = {
            'conteo_palabras': {},  
            'total_palabras': 0,    
            'num_documentos': 0      
        }
    
    conteo_clase = contar_palabras(archivos)
    
    for palabra, freq in conteo_clase.items():
        modelo['vocabulario'].add(palabra)
        
        if palabra in modelo['clases'][clase]['conteo_palabras']:
            modelo['clases'][clase]['conteo_palabras'][palabra] += freq
        else:
            modelo['clases'][clase]['conteo_palabras'][palabra] = freq
        modelo['clases'][clase]['total_palabras'] += freq
    
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
    if modelo is None or not modelo['clases']:
        return ''
    palabras = extraer_palabras(archivo)
    
    if not palabras:
        clase_mayor = ''
        max_docs = 0
        for clase in modelo['clases']:
            num_docs = modelo['clases'][clase]['num_documentos']
            if num_docs > max_docs:
                max_docs = num_docs
                clase_mayor = clase
        return clase_mayor
    
    # Tamaño del vocabulario (diccionario global)
    V = len(modelo['vocabulario'])
    
    # Calcular probabilidades para cada clase
    mejor_clase = None
    mejor_log_prob = float('-inf')
    
    for clase, info_clase in modelo['clases'].items():
        # P(y) = número de documentos de la clase / total de documentos
        # Este es el PRIOR
        prob_clase = info_clase['num_documentos'] / modelo['total_documentos']
        
        # Usar logaritmos para evitar underflow numérico
        # log(P(y|documento)) = log(P(y)) + Σ log(P(palabra|y))
        log_prob = log(prob_clase)
        
        # nj: total de posiciones de palabras en documentos de clase y
        total_palabras_clase = info_clase['total_palabras']
        conteo_palabras_clase = info_clase['conteo_palabras']
        
        # Para cada palabra en el documento
        for palabra in palabras:
            # nk: número de veces que palabra wk aparece en clase cj
            conteo_palabra = conteo_palabras_clase.get(palabra, 0)
            
            # Laplace Smoothing (m-estimate):
            # P(wk|cj) = (nk + K) / (nj + K * |Vocabulary|)
            # 
            # K es el "equivalent sample size"
            # Si K=0: no hay smoothing (puede dar prob=0)
            # Si K=1: Laplace smoothing estándar (add-one smoothing)
            # Si K>1: más peso a la distribución uniforme
            K_usado = max(K, 1)  # Mínimo 1 para evitar log(0)
            
            prob_palabra_dado_clase = (conteo_palabra + K_usado) / (total_palabras_clase + K_usado * V)
            
            # Agregar log de la probabilidad (LIKELIHOOD)
            log_prob += log(prob_palabra_dado_clase)
        
        # Actualizar mejor clase si esta tiene mayor probabilidad posterior
        if log_prob > mejor_log_prob:
            mejor_log_prob = log_prob
            mejor_clase = clase
    
    return mejor_clase if mejor_clase else ''
