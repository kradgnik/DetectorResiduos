import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
from pathlib import Path


@st.cache_resource
def load_model():
    model_path = Path("best.pt")
    if not model_path.exists():
        raise FileNotFoundError(
            "No se encontró el archivo 'best.pt' en el directorio actual. "
            "Súbelo a tu repositorio de GitHub junto con este archivo app.py."
        )
    return YOLO(str(model_path))


def main():
    st.title("Detección con YOLO (best.pt)")
    st.write("App creada a partir de tu notebook original con cámara web, adaptada para Streamlit.")

    model = load_model()

    modo = st.radio("¿Qué quieres procesar?", ["Imagen", "Video"], horizontal=True)

    if modo == "Imagen":
        archivo_img = st.file_uploader("Sube una imagen", type=["jpg", "jpeg", "png"])
        if archivo_img is not None:
            imagen = Image.open(archivo_img).convert("RGB")
            st.image(imagen, caption="Imagen original", use_column_width=True)

            with st.spinner("Detectando..."):
                resultados = model(imagen)
                imagen_annot = resultados[0].plot()

            st.image(imagen_annot, caption="Resultados YOLO", use_column_width=True)

    else:  # Video
        archivo_video = st.file_uploader("Sube un video", type=["mp4", "avi", "mov", "mkv"])
        if archivo_video is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(archivo_video.read())
            tfile.flush()

            st.info("Procesando video. Esto puede tardar un poco según la duración.")

            cap = cv2.VideoCapture(tfile.name)
            stframe = st.empty()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                resultados = model(frame)
                frame_annot = resultados[0].plot()  # BGR
                stframe.image(frame_annot, channels="BGR", use_column_width=True)

            cap.release()
            st.success("Procesamiento terminado.")


if __name__ == "__main__":
    main()
