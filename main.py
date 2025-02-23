from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],  # Métodos permitidos
    allow_headers=["*"],  # Cabeceras permitidas
)

# Carga la matriz de recomendaciones del disco
matriz_recomendaciones_long = pd.read_pickle("matriz_recomendaciones_long.pkl")

@app.get("/recomendaciones/{item_id}")
async def hacer_recomendacion(item_id: int, n: int = 3):
    # Verifica que el item exista en la matriz
    if item_id in matriz_recomendaciones_long['id1'].unique():
        # Filtra donde 'id1' sea igual al item proporcionado
        recomendaciones = matriz_recomendaciones_long[matriz_recomendaciones_long['id1'] == item_id]
        
        # Ordena por similitud de manera descendente y selecciona los primeros n resultados
        recomendaciones = recomendaciones.sort_values(by='similitud', ascending=False).head(n)
        
        return recomendaciones.to_dict(orient="records")
    else:
        raise HTTPException(status_code=404, detail=f"Error: El id {item_id} no se encuentra en las columnas del DataFrame.")