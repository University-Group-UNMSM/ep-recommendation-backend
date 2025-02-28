
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# IMPORTACIÓN DE DATOS
df = pd.read_csv("MOCK_DATA.csv")
df.dropna(subset=["image"], inplace=True)  # Elimina NaN en columna image desde el principio
df.rename(columns={'ratin': 'rating'}, inplace=True)  # Renombrar review/score
df['rating'] = df['rating'].astype(int)  # Cambiar el tipo de la columna review

# PREPARACIÓN DE DATOS
# Verificación de valores de review y visualización de los primeros datos
print(df['rating'].value_counts().sort_index())
print(df.head())

# Creamos una matriz item-usuario solo con los datos necesarios
iu = df.pivot_table(index='id_project', columns='id_user', values='rating', fill_value=0)

# Conservar información de imagen para cada Id
images = df[['id_project', 'image']].drop_duplicates().set_index('id_project')

# CÁLCULO DE LA SIMILITUD DE ITEMS
similitud_items = cosine_similarity(iu.to_numpy())

# CREACIÓN DE UNA MATRIZ DE RECOMENDACIONES
matriz_recomendaciones = pd.DataFrame(similitud_items, index=iu.index, columns=iu.index)

# Convertir el DataFrame de una matriz a un formato largo
matriz_recomendaciones_long = matriz_recomendaciones.stack().rename_axis(['id1_project', 'id2_project']).reset_index(name='similitud')
matriz_recomendaciones_long = matriz_recomendaciones_long[matriz_recomendaciones_long['id1_project'] != matriz_recomendaciones_long['id2_project']]
matriz_recomendaciones_long = matriz_recomendaciones_long[matriz_recomendaciones_long['id1_project'] < matriz_recomendaciones_long['id2_project']]

# Unir las imágenes de 'id2'
matriz_recomendaciones_long = matriz_recomendaciones_long.join(images, on='id2_project')

# GUARDA LA MATRIZ DE RECOMENDACION A DISCO
matriz_recomendaciones_long.to_pickle("matriz_recomendaciones_long.pkl")



