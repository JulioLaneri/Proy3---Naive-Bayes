#
# Universidad Nacional de Itapua
# Inteligencia Artificial - 7mo Semestre
#
# Proyecto#3: Clasificador de Bayes
#
# Autor: Amin Mansuri modificado por Wildo Monges
#
# NO ES NECESARIO MODIFICAR ESTE ARCHIVO TODOS TUS CAMBIOS DEBEN SER
# EN NAIVE_BAYES.PY. USAREMOS LA VERSION ORIGINAL PARA CORREGIR NO TU VERSION.
#

from math import floor
from os import listdir
from os.path import isfile,join

import naive_bayes
import random


LIMITE_K = 10

dirs = ('comp.os.ms-windows.misc','rec.sport.baseball',	'talk.politics.misc')


# Aqui guardamos nuestro modelo del clasificador
modelo = None
k = 0
max_correctos = 0
for K in range(LIMITE_K) :

  # Fase de entrenamiento
  #
  for d in dirs:

    # obtener nombres de archivos
    path = join('datos','entrenamiento',d)
    files = [ join(path,f)  for f in listdir(path) if isfile(join(path,f)) ]

    # entrenar modelo
    modelo = naive_bayes.entrenar(modelo, files, d, K)


  # Fase de validacion cruzada
  #

  correctos = 0
  al_correctos = 0
  for d in dirs :
 
    # obtener nombres de archivos
    path = join('datos','validacion', d)
    files = [ join(path,f)  for f in listdir(path) if isfile(join(path,f)) ]

    # para cada archivo
    for fn in files :

      # probar clasificador
      clase = naive_bayes.clasificar(modelo, fn, K)
      if clase == d :
        correctos += 1

      # probar al azar (para comparar)
      aclase = dirs[random.randint(0,len(dirs)-1)]
      if aclase == d:
        al_correctos += 1
  
  print('K = ', K, ' Correctos: ', correctos, ' Azar: ', al_correctos)
    
  if correctos > max_correctos :
      k = K 


# Ahora testear si esta bien
# ESTO SOLO LO HARA EL PROFE
#
testeo = False
if testeo :
  for d in dirs:

    # obtener nombres de archivos de entrenamiento
    path = join('datos','entrenamiento',d)
    entrenamiento = [ join(path,f) for f in listdir(path) if isfile(join(path,f)) ]
    
    # obtener nombres de archivos de validacion
    path = join('datos','validacion',d)
    validacion = [ join(path,f) for f in listdir(path) if isfile(join(path,f)) ]
  
    # entrenar modelo con todo
    modelo = naive_bayes.entrenar(entrenamiento + validacion, d, k)


  correctos = 0
  al_correctos = 0
  for d in dirs :
 
    # obtener nombres de archivos
    path = join('sol','test', d)
    files = [ f for f in listdir(path) if isfile(join(path,f)) ]

    # para cada archivo
    for fn in files :

      # probar clasificador
      clase = naive_bayes.clasificar(modelo, fn, K)
      if clase == d :
        correctos += 1

      # probar al azar (para comparar)
      aclase = dirs[random.randint(0,len(dirs)-1)]
      if aclase == d:
        al_correctos += 1
  
  
  print('RESULTADO FINAL ->  Correctos: ', correctos, ' Azar: ', al_correctos)
