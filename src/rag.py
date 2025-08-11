import os
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import Document
from ocr import tool_ocr

dotenv.load_dotenv()

INDEX_PATH = "/home/gacoelho/Documents/agente_emisssao_documento/doc"

# Extrai texto de documentos na pasta via OCR e retorna lista de Documentos
def extrair_documentos_por_ocr(pasta_docs: str) -> list[Document]:
    documentos = []
    for arquivo in os.listdir(pasta_docs):
        caminho = os.path.join(pasta_docs, arquivo)
        if os.path.isfile(caminho):
            resultado = tool_ocr(caminho)
            texto = resultado.get("texto_extraido", "").strip()
            if texto:
                doc = Document(
                    page_content=texto,
                    metadata={"arquivo": arquivo, "num_paginas": resultado.get("num_paginas", 0)}
                )
                documentos.append(doc)
    return documentos

# Permite ao usuário escolher entre múltiplos documentos relevantes
def escolher_documento_opcoes(opcoes: list[Document]) -> Document:
    print("Encontrei mais de uma opção relevante para sua pergunta. Por favor, escolha uma:")

    for i, doc in enumerate(opcoes, 1):
        resumo = doc.page_content[:100].replace("\n", " ")  # Mostrar os primeiros 100 caracteres
        print(f"{i}: {resumo}...")

    while True:
        escolha = input(f"Digite um número entre 1 e {len(opcoes)}: ").strip()
        if escolha.isdigit() and 1 <= int(escolha) <= len(opcoes):
            return opcoes[int(escolha) - 1]
        else:
            print("Opção inválida, tente novamente.")

# Cria índice FAISS a partir dos documentos extraídos via OCR
def criar_indice_faiss(embeddings):
    print("[INFO] Criando índice FAISS via OCR dos documentos na pasta...")

    docs = extrair_documentos_por_ocr(INDEX_PATH)

    if not docs:
        print("[WARN] Nenhum documento extraído via OCR para criar índice.")
        return

    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local(INDEX_PATH)
    print(f"[INFO] Índice FAISS criado e salvo em {INDEX_PATH}")

# Executa o pipeline RAG com Azure OpenAI e FAISS
def executar_rag(pergunta: str) -> str:
    # Embeddings via Azure
    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        model=os.getenv("EMBEDDINGS_MODEL_NAME")  # ou nome do deployment
    )

    # Criar índice se não existir
    index_faiss = os.path.join(INDEX_PATH, "index.faiss")
    index_pkl = os.path.join(INDEX_PATH, "index.pkl")
    if not (os.path.exists(index_faiss) and os.path.exists(index_pkl)):
        os.makedirs(INDEX_PATH, exist_ok=True)
        criar_indice_faiss(embeddings)

    # Carregar vetor FAISS
    vectorstore = FAISS.load_local(
        INDEX_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM via Azure Chat
    llm = AzureChatOpenAI(
        api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
        api_key=os.getenv("AZURE_OPENAI_KEY")
    )

    # Buscar contexto e gerar resposta
    docs = retriever.invoke(pergunta)

    # Se tiver mais de uma opção, pedir para o usuário escolher
    if len(docs) > 1:
        doc_escolhido = escolher_documento_opcoes(docs)
        contexto = doc_escolhido.page_content
    elif docs:
        contexto = docs[0].page_content
    else:
        contexto = ""

    prompt = f"Pergunte ao usário qual destes documento ele quer\n\nContexto:\n{contexto}\n\nPergunta: {pergunta}\nResposta:"
    resposta = llm.invoke(prompt)

    return resposta


if __name__ == "__main__":
    pergunta_usuario = input("Você: ")
    resposta = executar_rag(pergunta_usuario)
    print("\nBot:", resposta.content)
