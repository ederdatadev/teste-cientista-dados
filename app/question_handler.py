import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/product_order_details.csv")

def humanizar_resposta(resposta_obj):
    return resposta_obj

def top_subcategorias_cartao():
    try:
        filtro = df[(df["Category"] == "Electronics") & (df["PaymentMode"] == "Credit Card")]
        return {"top_3_subcategorias_cartao": filtro["Sub-Category"].value_counts().head(3).to_dict()}
    except Exception as e:
        return {"erro": str(e)}

def categorias_mais_vendidas():
    try:
        resultado = df.groupby("Category")["Quantity"].sum().sort_values(ascending=False).to_dict()
        return {"categorias_mais_vendidas": {k: int(v) for k, v in resultado.items()}}
    except Exception as e:
        return {"erro": str(e)}

def subcategorias_maior_valor_medio():
    try:
        df["Valor Médio"] = df["Amount"].astype(float) / df["Quantity"].astype(float)
        resultado = df.groupby("Sub-Category")["Valor Médio"].mean().sort_values(ascending=False).to_dict()
        return {"subcategorias_maior_valor_medio": {k: round(v, 2) for k, v in resultado.items()}}
    except Exception as e:
        return {"erro": str(e)}

def metodos_pagamento():
    try:
        return {"top_2_metodos_pagamento": df["PaymentMode"].value_counts().head(2).to_dict()}
    except Exception as e:
        return {"erro": str(e)}

def lucro_medio_categoria():
    try:
        resultado = df.groupby("Category")["Profit"].mean().to_dict()
        return {"lucro_medio_por_categoria": {k: round(v, 2) for k, v in resultado.items()}}
    except Exception as e:
        return {"erro": str(e)}

def produtos_maior_prejuizo():
    try:
        df_prejuizo = df[df["Profit"] < 0].sort_values(by="Profit", ascending=True).head(3)
        return {"top_3_produtos_maior_prejuizo": df_prejuizo[["Sub-Category", "Profit"]].to_dict(orient="records")}
    except Exception as e:
        return {"erro": str(e)}

def subcategorias_mais_pedidos_unicos():
    try:
        return {"subcategorias_mais_pedidos_unicos": df["Sub-Category"].value_counts().to_dict()}
    except Exception as e:
        return {"erro": str(e)}

def subcategoria_mais_lucrativa():
    try:
        resultado = df.groupby("Sub-Category")["Profit"].sum()
        return {"subcategoria_mais_lucrativa": {"subcategory": resultado.idxmax(), "total_profit": int(resultado.max())}}
    except Exception as e:
        return {"erro": str(e)}

def ticket_medio_pagamento():
    try:
        df["Ticket Medio"] = df["Amount"].astype(float) / df["Quantity"].astype(float)
        resultado = df.groupby("PaymentMode")["Ticket Medio"].mean().to_dict()
        return {"ticket_medio_pagamento": {k: round(v, 2) for k, v in resultado.items()}}
    except Exception as e:
        return {"erro": str(e)}

def resposta_padrao():
    return {"resposta": "Desculpe, não tenho essa informação no momento."}

question_map = {
    "subcategorias_cartao": top_subcategorias_cartao,
    "categorias_mais_vendidas": categorias_mais_vendidas,
    "subcategorias_valor_medio": subcategorias_maior_valor_medio,
    "metodos_pagamento": metodos_pagamento,
    "lucro_medio_categoria": lucro_medio_categoria,
    "produtos_maior_prejuizo": produtos_maior_prejuizo,
    "subcategorias_mais_pedidos_unicos": subcategorias_mais_pedidos_unicos,
    "subcategoria_mais_lucrativa": subcategoria_mais_lucrativa,
    "ticket_medio_pagamento": ticket_medio_pagamento,
    "resposta_padrao": resposta_padrao
}

def encontrar_categoria(pergunta_corrigida):
    categorias = list(question_map.keys())

    frases_treinamento = [
        "subcategorias cartao credito pagamento",
        "categorias mais vendidas quantidade",
        "subcategorias maior valor medio item",
        "metodos pagamento mais utilizados",
        "lucro medio categoria produto",
        "produtos maior prejuizo negativo",
        "subcategoria mais pedidos unicos",
        "subcategoria mais lucro total",
        "ticket medio metodo pagamento"
    ]

    vectorizer = TfidfVectorizer()
    matriz_tfidf = vectorizer.fit_transform(frases_treinamento)

    pergunta_tfidf = vectorizer.transform([pergunta_corrigida])

    similaridades = cosine_similarity(pergunta_tfidf, matriz_tfidf)[0]

    melhor_indice = similaridades.argmax()
    melhor_categoria = categorias[melhor_indice]

    if similaridades[melhor_indice] < 0.8:
        return "resposta_padrao"

    return melhor_categoria
