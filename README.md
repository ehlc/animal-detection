# 🦓 Animal Detection

API ligera desarrollada con **FastAPI** para procesar imágenes de manadas densas y obtener predicciones sobre la cantidad de animales presentes.

## 🚀 Descripción

Esta API expone un único endpoint `/predict` que permite enviar una imagen en formato JPG o JPEG. La imagen se sube a un bucket S3 y se simula una solicitud de inferencia para retornar un JSON con una estructura de conteo por clase y confianza promedio.


## 🧪 Requisitos

Instala las dependencias ejecutando:

```bash
pip install -r requirements.txt
```

## 📦 Estructura del repositorio

```
📁 animal-detection-api/
├── app.py               # Código principal de la API
├── requirements.txt     # Dependencias del proyecto
├── .env                 # Variables de entorno (no incluida en el repo)
└── README.md            # Este archivo
```

## 🔌 Endpoint disponible

### POST `/predict`

- **Entrada**: imagen JPG, JPEG o PNG (como `multipart/form-data`)
- **Respuesta**: JSON con predicción simulada

#### Ejemplo de uso con `curl`:

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@/ruta/a/la/imagen.jpg"
```

#### Ejemplo de respuesta:

```json
{
  "url": "",
  "inference": [
    {"class_id": 0, "class_name": "Clase 0", "count": 15, "avg_confidence": 2.03},
    {"class_id": 2, "class_name": "Clase 2", "count": 107, "avg_confidence": 5.34},
    {"class_id": 4, "class_name": "Clase 4", "count": 29, "avg_confidence": 4.13},
    {"class_id": 5, "class_name": "Clase 5", "count": 9, "avg_confidence": 1.88}
  ]
}
```

## ⚙️ Ejecución local

```bash
uvicorn app:app --reload --port 8000
```


## 📄 Licencia

MIT License. Consulta el archivo `LICENSE` si está presente.
