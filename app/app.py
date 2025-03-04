from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.question_handler import question_map, encontrar_categoria
from symspellpy import SymSpell
import os

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

sym_spell = SymSpell(max_dictionary_edit_distance=2)

dict_path = "portuguese_words.txt"

if os.path.exists(dict_path):
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1, separator="\n")
else:
    print(f"⚠️ Aviso: Dicionário {dict_path} não encontrado. Correção ortográfica desativada.")

def corrigir_texto(texto: str) -> str:
    if sym_spell.words:
        sugestao = sym_spell.lookup_compound(texto, max_edit_distance=2)
        return sugestao[0].term if sugestao else texto
    return texto

def processar_pergunta(pergunta: str) -> dict:
    try:
        print(f"🔍 Pergunta recebida: {pergunta}")

        pergunta_corrigida = corrigir_texto(pergunta)
        print(f"✅ Pergunta corrigida: {pergunta_corrigida}")

        chave_encontrada = encontrar_categoria(pergunta_corrigida)

        if chave_encontrada and chave_encontrada in question_map:
            resposta = question_map[chave_encontrada]()
            print(f"✅ Correspondência encontrada: {chave_encontrada}")
            return {
                "pergunta_original": pergunta,
                "pergunta_corrigida": pergunta_corrigida,
                "categoria_predita": chave_encontrada,
                "resposta": resposta
            }

        print("⚠️ Nenhuma correspondência exata encontrada.")
        return {
            "erro": "Não encontrei uma resposta para essa pergunta. Tente reformular.",
            "pergunta_recebida": pergunta
        }

    except Exception as e:
        print(f"❌ Erro ao processar a pergunta: {str(e)}")
        return {
            "erro": "Ocorreu um erro ao processar a pergunta.",
            "detalhes": str(e)
        }

@app.get("/")
def home():
    return {"mensagem": "API de NLP para responder perguntas sobre pedidos!"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    try:
        pergunta = request.question.strip()
        if not pergunta:
            raise HTTPException(status_code=400, detail="A pergunta não pode estar vazia.")

        return processar_pergunta(pergunta)

    except HTTPException as http_err:
        raise http_err

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
