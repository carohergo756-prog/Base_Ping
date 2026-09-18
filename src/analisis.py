import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Creación de la carpeta de salidas como respaldo 
os.makedirs("outputs/resultados", exist_ok=True)

# Carga de la base de datos
df = pd.read_csv("data/penguins_lter.csv")

print("---------------------")
print("Dimensiones:", df.shape)
print("\nColumnas encontradas:")
print(df.columns.tolist())
print("\nValores nulos:")
print(df.isnull().sum())
print("\nDuplicados:", df.duplicated().sum())

# Limpieza y nomalización de nombres / simbolos
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

columnas_base = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g']
df = df.dropna(subset=columnas_base).copy()

# Clasificación de Sexo en Femenino y Masculino
df = df[df['sex'].isin(['MALE', 'FEMALE'])].copy()
# Opcional: traducir a español para visualización
df['sex'] = df['sex'].replace({'MALE': 'MASCULINO', 'FEMALE': 'FEMENINO'})

# Nombre de la especie 
df['especie'] = df['species'].apply(lambda x: x.split()[0])

# Variables
df['proporcion_pico'] = df['culmen_length_mm'] / df['culmen_depth_mm']
df['peso_kg'] = df['body_mass_g'] / 1000

print("\n Datos nuevos")
print("Filas listas:", df.shape[0])


print("-----------------------------------------")

# Análisis y preguntas
print("\n1.- Peso promedio (kg) por especie y sexo:")
print(df.groupby(['especie', 'sex'])['peso_kg'].mean().round(2))

print("\n2.- Distribucion por isla:")
print(pd.crosstab(df['especie'], df['island']))

print("\n3.- Proporcion de pico por especie:")
print(df.groupby('especie')['proporcion_pico'].mean().round(2))

print("\n4.- Correlacion aleta vs peso:")
corr = df['flipper_length_mm'].corr(df['peso_kg'])
print("Correlacion:", round(corr, 3))

# Gráficas / Visualizaciones
sns.set_theme(style="whitegrid")

# Grafica acerca del pico
plt.figure(figsize=(7, 5))
sns.scatterplot(data=df, x='culmen_length_mm', y='culmen_depth_mm', hue='especie')
plt.title("Largo vs Profundidad del Pico")
plt.xlabel("Largo (mm)")
plt.ylabel("Profundidad (mm)")
plt.savefig("outputs/resultados/grafico_pico.png", bbox_inches='tight')
plt.close()

# Grafica del peso por sexo
plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='especie', y='peso_kg', hue='sex')
plt.title("Peso por Especie y Sexo")
plt.xlabel("Especie")
plt.ylabel("Peso (kg)")
plt.savefig("outputs/resultados/grafico_peso.png", bbox_inches='tight')
plt.close()

# Grafica de cantidad de especies por isla
plt.figure(figsize=(7, 5))
pd.crosstab(df['especie'], df['island']).plot(kind='bar', stacked=True)
plt.title("Poblacion por Isla")
plt.xlabel("Especie")
plt.ylabel("Cantidad")
plt.xticks(rotation=0)
plt.savefig("outputs/resultados/grafico_islas.png", bbox_inches='tight')
plt.close()

