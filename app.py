# =========================================================
# App Streamlit: OCR + NLP para noticias de periodicos peruanos
# Curso: Ciencia de Datos II (UNALM)
# =========================================================
import json
import time
import re
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE = Path(__file__).parent
EXT_VALIDAS = ["png", "jpg", "jpeg"]
VARIANTES = {
    "Redimensionada (color)": "2. Redimensionada",
    "Grises + CLAHE": "5. Contraste (CLAHE)",
    "Binarizada": "6. Binarizada",
}


# ---------------------------------------------------------
# Carga de recursos (se cargan una sola vez)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de clasificacion...")
def cargar_modelo():
    return joblib.load(BASE / "modelo_tema.joblib")


@st.cache_resource
def cargar_stopwords():
    return set((BASE / "stopwords_es.txt").read_text(encoding="utf-8").split())


@st.cache_resource(show_spinner="Cargando spaCy...")
def cargar_spacy():
    try:
        import spacy
        return spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except Exception:
        return None


@st.cache_resource(show_spinner="Cargando motor OCR (la primera vez tarda)...")
def cargar_ocr():
    try:
        import easyocr
        return "EasyOCR", easyocr.Reader(["es"], gpu=False)
    except Exception:
        pass
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        return "Tesseract", pytesseract
    except Exception:
        return None, None


@st.cache_data
def cargar_metricas():
    ruta = BASE / "metricas.json"
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


# ---------------------------------------------------------
# Preprocesamiento de imagenes (igual que en el cuaderno)
# ---------------------------------------------------------
def redimensionar(img, lado_objetivo=1800):
    h, w = img.shape[:2]
    escala = lado_objetivo / max(h, w)
    interp = cv2.INTER_CUBIC if escala > 1 else cv2.INTER_AREA
    return cv2.resize(img, None, fx=escala, fy=escala, interpolation=interp)


def preprocesar(img_bgr, lado=1800):
    etapas = {"1. Original": img_bgr}
    redim = redimensionar(img_bgr, lado)
    etapas["2. Redimensionada"] = redim
    gris = cv2.cvtColor(redim, cv2.COLOR_BGR2GRAY)
    etapas["3. Escala de grises"] = gris
    sin_ruido = cv2.fastNlMeansDenoising(gris, None, h=10, templateWindowSize=7, searchWindowSize=21)
    etapas["4. Sin ruido"] = sin_ruido
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contraste = clahe.apply(sin_ruido)
    etapas["5. Contraste (CLAHE)"] = contraste
    binaria = cv2.adaptiveThreshold(contraste, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 15)
    etapas["6. Binarizada"] = binaria
    return etapas


def ocr_texto(img, motor, lector):
    if motor == "EasyOCR":
        return "\n".join(lector.readtext(img, detail=0, paragraph=True))
    if motor == "Tesseract":
        try:
            idioma = "spa" if "spa" in lector.get_languages() else "eng"
        except Exception:
            idioma = "eng"
        return lector.image_to_string(img, lang=idioma)
    raise RuntimeError("No hay un motor OCR disponible.")


# ---------------------------------------------------------
# Limpieza de texto (igual que en el cuaderno)
# ---------------------------------------------------------
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúñü\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar_sin_stopwords(texto_limpio, stop):
    return [t for t in texto_limpio.split() if t not in stop and len(t) > 2]


def lematizar(tokens, nlp):
    if nlp is None:
        return " ".join(tokens)
    doc = nlp(" ".join(tokens))
    return " ".join(t.lemma_ for t in doc if t.lemma_.isalpha() and len(t.lemma_) > 2)


# ---------------------------------------------------------
# Analisis completo de una imagen
# ---------------------------------------------------------
def analizar_imagen(img_bgr, variante, motor, lector, modelo, stop, nlp):
    t0 = time.time()
    etapas = preprocesar(img_bgr)
    texto_ocr = ocr_texto(etapas[VARIANTES[variante]], motor, lector)
    limpio = limpiar_texto(texto_ocr)
    tokens = tokenizar_sin_stopwords(limpio, stop)
    proc = lematizar(tokens, nlp)

    resultado = {
        "texto_ocr": texto_ocr,
        "texto_limpio": limpio,
        "n_tokens": len(tokens),
        "palabras_clave": pd.Series(tokens).value_counts().head(10) if tokens else pd.Series(dtype=int),
        "tema": None,
        "confianza": None,
        "probabilidades": None,
        "segundos": 0.0,
    }
    if len(tokens) >= 5:
        proba = pd.Series(modelo.predict_proba([proc])[0], index=modelo.classes_).sort_values(ascending=False)
        resultado["tema"] = proba.index[0]
        resultado["confianza"] = float(proba.iloc[0])
        resultado["probabilidades"] = proba
    resultado["segundos"] = time.time() - t0
    return resultado


def decodificar(datos):
    return cv2.imdecode(np.frombuffer(datos, dtype=np.uint8), cv2.IMREAD_COLOR)


# ---------------------------------------------------------
# Interfaz
# ---------------------------------------------------------
def main():
    st.set_page_config(page_title="OCR + NLP de noticias peruanas", page_icon="📰", layout="wide")
    st.title("📰 Clasificador de noticias de periódicos peruanos")
    st.caption("Extrae el texto de una imagen con OCR y predice su tema "
               "(política, policial, espectáculo, economía o deporte) con NLP.")

    modelo = cargar_modelo()
    stop = cargar_stopwords()
    nlp = cargar_spacy()
    motor, lector = cargar_ocr()
    metricas = cargar_metricas()

    # ---- barra lateral ----
    with st.sidebar:
        st.header("Configuración")
        variante = st.selectbox("Preprocesamiento antes del OCR", list(VARIANTES),
                                help="En la validación, 'Redimensionada (color)' obtuvo el menor error de OCR.")
        mostrar_etapas = st.checkbox("Mostrar etapas del preprocesamiento", value=False)
        st.divider()
        st.markdown(f"**Motor OCR:** {motor or 'no disponible'}")
        st.markdown(f"**Lematización spaCy:** {'sí' if nlp is not None else 'no disponible'}")
        if metricas:
            st.divider()
            st.subheader("Desempeño del modelo")
            st.markdown(f"Validación cruzada ({metricas.get('n_documentos', '?')} noticias)")
            st.metric("Accuracy", f"{metricas.get('accuracy_cv', 0):.3f}")
            st.metric("F1 macro", f"{metricas.get('f1_macro_cv', 0):.3f}")
            ocr = metricas.get("ocr", {})
            if ocr:
                st.markdown(f"OCR (mediana): CER {ocr.get('CER_mediana')} · WER {ocr.get('WER_mediana')}")

    if motor is None:
        st.error("No hay ningún motor OCR disponible en este entorno (EasyOCR o Tesseract).")
        st.stop()

    # ---- entrada de imagenes ----
    archivos = st.file_uploader("Sube una o varias imágenes de noticias", type=EXT_VALIDAS,
                                accept_multiple_files=True)
    entradas = [(a.name, a.getvalue()) for a in archivos]

    ejemplos = sorted(p for p in (BASE / "ejemplos").glob("*") if p.suffix.lower().lstrip(".") in EXT_VALIDAS)
    if not entradas and ejemplos:
        elegido = st.selectbox("…o prueba con una imagen de ejemplo", ["(ninguna)"] + [p.name for p in ejemplos])
        if elegido != "(ninguna)":
            ruta = BASE / "ejemplos" / elegido
            entradas = [(ruta.name, ruta.read_bytes())]

    if not entradas:
        st.info("Sube una imagen (PNG o JPG) para empezar.")
        return

    # ---- procesamiento ----
    filas = []
    for nombre, datos in entradas:
        img = decodificar(datos)
        if img is None:
            st.warning(f"No se pudo leer la imagen: {nombre}")
            continue

        with st.spinner(f"Procesando {nombre}..."):
            r = analizar_imagen(img, variante, motor, lector, modelo, stop, nlp)

        with st.expander(f"{nombre}  →  {r['tema'] or 'texto insuficiente'}", expanded=len(entradas) == 1):
            c1, c2 = st.columns([1, 2])
            with c1:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=nombre)
            with c2:
                if r["tema"]:
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Tema predicho", r["tema"].capitalize())
                    m2.metric("Confianza", f"{r['confianza']:.0%}")
                    m3.metric("Tiempo", f"{r['segundos']:.1f} s")
                else:
                    st.warning("El OCR extrajo muy poco texto para clasificar la noticia.")

                t1, t2, t3, t4 = st.tabs(["Texto extraído", "Texto limpio", "Análisis NLP", "Preprocesamiento"])
                with t1:
                    st.text_area("Resultado del OCR", r["texto_ocr"], height=250, key=f"ocr_{nombre}")
                with t2:
                    st.text_area("Texto en minúsculas, sin símbolos", r["texto_limpio"], height=250,
                                 key=f"limpio_{nombre}")
                    st.caption(f"{r['n_tokens']} palabras útiles tras quitar stopwords")
                with t3:
                    if r["probabilidades"] is not None:
                        st.markdown("**Probabilidad por tema**")
                        st.bar_chart(r["probabilidades"])
                    if len(r["palabras_clave"]):
                        st.markdown("**Palabras más frecuentes**")
                        st.bar_chart(r["palabras_clave"])
                with t4:
                    if mostrar_etapas:
                        etapas = preprocesar(img)
                        for nom, im in etapas.items():
                            im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im
                            st.image(im_rgb, caption=nom)
                    else:
                        st.caption("Activa 'Mostrar etapas del preprocesamiento' en la barra lateral.")

        fila = {"archivo": nombre, "tema": r["tema"], "confianza": r["confianza"],
                "texto_ocr": r["texto_ocr"], "texto_limpio": r["texto_limpio"]}
        if r["probabilidades"] is not None:
            for cat, p in r["probabilidades"].items():
                fila[f"prob_{cat}"] = round(float(p), 4)
        filas.append(fila)

    # ---- descarga de resultados ----
    if filas:
        st.subheader("Resultados")
        tabla = pd.DataFrame(filas)
        st.dataframe(tabla[["archivo", "tema", "confianza"]])
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Descargar resultados (CSV)", tabla.to_csv(index=False).encode("utf-8-sig"),
                           file_name="resultados_ocr_nlp.csv", mime="text/csv")
        texto_total = "\n\n".join(f"=== {f['archivo']} ({f['tema']}) ===\n{f['texto_ocr']}" for f in filas)
        c2.download_button("⬇️ Descargar texto extraído (TXT)", texto_total.encode("utf-8"),
                           file_name="texto_extraido.txt", mime="text/plain")


if __name__ == "__main__":
    main()
