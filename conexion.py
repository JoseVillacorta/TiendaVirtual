import mysql.connector
import pandas as pd
from sqlalchemy import create_engine
from mlxtend.frequent_patterns import apriori, association_rules

# Conexión a la base de datos MySQL en Clever Cloud
db_url = "mysql+pymysql://uaa788fiv7nrorft:upZFc2COJuqH4oh8mKIO@bspmuf4r5rq7bcezlile-mysql.services.clever-cloud.com/bspmuf4r5rq7bcezlile"
engine = create_engine(db_url)

# Extracción de datos de transacciones
query = """
SELECT CartItems.cart_id, Products.nombre 
FROM CartItems 
JOIN Cart ON CartItems.cart_id = Cart.cart_id
JOIN Products ON CartItems.product_id = Products.product_id
"""
df = pd.read_sql(query, con=engine)

# Convertir los datos a formato de una sola fila por transacción
basket = df.groupby(['cart_id', 'nombre'])['nombre'].count().unstack().reset_index().fillna(0).set_index('cart_id')

# Convertir a valores binarios
def encode_units(x):
    return 1 if x >= 1 else 0

basket = basket.applymap(encode_units)

# Aplicar el algoritmo Apriori con un soporte mínimo menor
frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)

# Verificar si se encontraron conjuntos frecuentes
if frequent_itemsets.empty:
    print("No se encontraron conjuntos frecuentes con el soporte especificado.")
else:
    # Generar reglas de asociación
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)

    # Traducir las columnas al español
    rules.rename(columns={
        'antecedents': 'Antecedentes',
        'consequents': 'Consecuentes',
        'antecedent support': 'Soporte de Antecedentes',
        'consequent support': 'Soporte de Consecuentes',
        'support': 'Soporte',
        'confidence': 'Confianza',
        'lift': 'Elevación',
        'leverage': 'Apalancamiento',
        'conviction': 'Convicción'
    }, inplace=True)

    # Mostrar las reglas
    print(rules)
