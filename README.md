# Clasificador de noticias de periódicos peruanos (OCR + NLP)

Proyecto del curso **Ciencia de Datos II – Procesamiento de Lenguaje Natural** (UNALM).

Aplicación en Streamlit que recibe la imagen de una noticia de un periódico peruano, extrae su texto con **OCR (EasyOCR)**,
lo limpia con técnicas de **NLP** (minúsculas, eliminación de símbolos, tokenización, stopwords y lematización con spaCy)
y predice su tema: política, policial, espectáculo, economía o deporte.

## Flujo
imagen → preprocesamiento (OpenCV) → OCR → limpieza del texto → TF-IDF + Regresión logística → tema

## Resultados
- Dataset propio: 82 imágenes de noticias etiquetadas en 5 categorías.
- Clasificación (validación cruzada estratificada de 5 particiones): accuracy 0.768 y F1 macro 0.746.
- OCR (mediana, 9 imágenes con transcripción de referencia): CER 0.197 y WER 0.5
  con la imagen solo redimensionada a color, que superó a las variantes con CLAHE y binarización.

## Archivos
- `app.py`: aplicación Streamlit.
- `modelo_tema.joblib`: modelo de clasificación entrenado.
- `stopwords_es.txt`: stopwords usadas en el entrenamiento.
- `metricas.json`: métricas mostradas en la app.
- `ejemplos/`: imágenes de prueba.
- `requirements.txt` y `packages.txt`: dependencias de Python y del sistema.

## Ejecutar localmente
```
pip install -r requirements.txt
streamlit run app.py
```

## Limitaciones
Dataset pequeño y desbalanceado (economía tiene pocas imágenes), pocas transcripciones de referencia para medir el OCR,
y el diseño en columnas de los periódicos puede alterar el orden de lectura del OCR.

## Integrantes
- Blácido Hernández, Pamela
- Condor Montes, Paola
- Chavez Paucar, Carla
