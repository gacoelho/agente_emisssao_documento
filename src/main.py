from langgraph import graph
from langgraph.graph import StateGraph, START, END
from dataclasses import dataclass, field
from typing import List, Dict, Any, Annotated
import os

@dataclass
class EstadoFluxo:
    indice_existe: bool = False
    tipo_documento: str = ""
    docs_extraidos: list = field(default_factory=list)
    resposta: str = ""
    pergunta_usuario: str = ""
    documentos_encontrados: List[Dict] = field(default_factory=list)
    documento_escolhido: Dict = field(default_factory=dict)
    contexto_rag: str = ""

def verificar_indice_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Verifica se o índice FAISS existe"""
    import os
    INDEX_PATH = "/home/gacoelho/Documents/agente_emisssao_documento/doc"
    index_faiss = os.path.join(INDEX_PATH, "index.faiss")
    index_pkl = os.path.join(INDEX_PATH, "index.pkl")

    estado.indice_existe = os.path.exists(index_faiss) and os.path.exists(index_pkl)
    print(f"[Verificar Índice] Índice existe? {estado.indice_existe}")
    return {"estado": estado}

def criar_indice_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Cria o índice FAISS se não existir"""
    print("[Criar Índice] Criando índice FAISS com documentos...")
    try:
        from rag import DocumentProcessor
        processor = DocumentProcessor()
        success = processor.criar_indice_faiss()
        estado.indice_existe = success
        return {"estado": estado}
    except Exception as e:
        print(f"[Erro] Falha ao criar índice: {e}")
        estado.indice_existe = False
        return {"estado": estado}

def coletar_pergunta_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Coleta a pergunta do usuário"""
    pergunta = input("👤 Digite sua pergunta sobre documentos: ").strip()
    estado.pergunta_usuario = pergunta
    print(f"[Pergunta] Usuário perguntou: {pergunta}")
    return {"estado": estado}

def buscar_documentos_rag_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Busca documentos relevantes usando RAG"""
    print(f"[RAG] Buscando documentos para: '{estado.pergunta_usuario}'")
    try:
        from rag import DocumentProcessor
        processor = DocumentProcessor()
        
        # Buscar documentos similares
        docs = processor.buscar_documentos_similares(estado.pergunta_usuario, max_results=5)
        
        if not docs:
            estado.resposta = "❌ Nenhum documento relevante encontrado para sua pergunta."
            return {"estado": estado}
        
        # Converter para formato mais simples
        documentos_info = []
        for doc in docs:
            doc_info = {
                "arquivo": doc.metadata.get("arquivo", "Desconhecido"),
                "chunk_id": doc.metadata.get("chunk_id", 0),
                "tipo_arquivo": doc.metadata.get("tipo_arquivo", "desconhecido"),
                "conteudo": doc.page_content[:300].replace("\n", " ").strip(),
                "documento_completo": doc
            }
            documentos_info.append(doc_info)
        
        estado.documentos_encontrados = documentos_info
        print(f"[RAG] Encontrados {len(documentos_info)} documentos relevantes")
        
        return {"estado": estado}
            
    except Exception as e:
        estado.resposta = f"❌ Erro ao buscar documentos: {str(e)}"
        return {"estado": estado}

def mostrar_documentos_encontrados_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Mostra os documentos encontrados para o usuário escolher"""
    print(f"\n🔍 Encontrei {len(estado.documentos_encontrados)} documentos relevantes:")
    print("=" * 80)
    
    for i, doc in enumerate(estado.documentos_encontrados, 1):
        arquivo = doc["arquivo"]
        chunk_id = doc["chunk_id"]
        tipo = doc["tipo_arquivo"]
        resumo = doc["conteudo"]
        
        print(f"{i}. 📄 {arquivo}")
        print(f"   📍 Chunk {chunk_id} | Tipo: {tipo}")
        print(f"   📝 {resumo}...")
        print()
    
    # Permitir escolha do usuário
    while True:
        try:
            escolha = input(f"🎯 Escolha um documento (1-{len(estado.documentos_encontrados)}) ou 'auto' para o mais relevante: ").strip()
            
            if escolha.lower() == 'auto':
                estado.documento_escolhido = estado.documentos_encontrados[0]
                print(f"✅ Documento escolhido automaticamente: {estado.documento_escolhido['arquivo']}")
                break
            
            if escolha.isdigit() and 1 <= int(escolha) <= len(estado.documentos_encontrados):
                idx = int(escolha) - 1
                estado.documento_escolhido = estado.documentos_encontrados[idx]
                print(f"✅ Documento escolhido: {estado.documento_escolhido['arquivo']}")
                break
            else:
                print("❌ Opção inválida, tente novamente.")
        except KeyboardInterrupt:
            print("\n👋 Operação cancelada")
            estado.documento_escolhido = estado.documentos_encontrados[0]
            break
    
    return {"estado": estado}

def executar_consulta_rag_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Executa a consulta RAG no documento escolhido"""
    print(f"[RAG] Executando consulta no documento: {estado.documento_escolhido['arquivo']}")
    
    try:
        from rag import DocumentProcessor
        processor = DocumentProcessor()
        
        # Executar RAG com o documento específico
        resposta = processor.executar_rag(estado.pergunta_usuario, max_results=1, auto_clarify=False)
        
        estado.resposta = resposta
        estado.contexto_rag = f"Documento consultado: {estado.documento_escolhido['arquivo']}"
        
        print(f"[RAG] Consulta executada com sucesso")
        return {"estado": estado}
        
    except Exception as e:
        estado.resposta = f"❌ Erro ao executar consulta RAG: {str(e)}"
        return {"estado": estado}

def apresentar_resultado_node(estado: EstadoFluxo) -> Dict[str, Any]:
    """Apresenta o resultado da consulta RAG"""
    print("\n" + "=" * 80)
    print("📋 RESULTADO DA CONSULTA")
    print("=" * 80)
    print(f"🔍 Pergunta: {estado.pergunta_usuario}")
    print(f"📄 Documento: {estado.documento_escolhido['arquivo']}")
    print(f"📍 Chunk: {estado.documento_escolhido['chunk_id']}")
    print("-" * 80)
    print(estado.resposta)
    print("=" * 80)
    
    return {"estado": estado}

def main():
    """Função principal com fluxo LangGraph simplificado"""
    print("🤖 Sistema de Consulta RAG - LangGraph")
    print("=" * 60)
    print("💡 Este sistema usa RAG para encontrar e consultar documentos reais")
    print("⚠️  IMPORTANTE: Não inventa informações - usa apenas documentos existentes")
    print("-" * 60)
    
    # Estado inicial
    estado = EstadoFluxo()
    
    # Verificar índice
    if not estado.indice_existe:
        print("[Verificar Índice] Verificando se o índice existe...")
        INDEX_PATH = "/home/gacoelho/Documents/agente_emisssao_documento/doc"
        index_faiss = os.path.join(INDEX_PATH, "index.faiss")
        index_pkl = os.path.join(INDEX_PATH, "index.pkl")
        estado.indice_existe = os.path.exists(index_faiss) and os.path.exists(index_pkl)
        
        if not estado.indice_existe:
            print("[Criar Índice] Criando índice FAISS...")
            try:
                from rag import DocumentProcessor
                processor = DocumentProcessor()
                success = processor.criar_indice_faiss()
                estado.indice_existe = success
            except Exception as e:
                print(f"[Erro] Falha ao criar índice: {e}")
                return
    
    print("✅ Sistema inicializado com sucesso!")
    print("🔍 O sistema criará automaticamente o índice FAISS se necessário")
    print("-" * 60)
    
    # Loop principal de consulta
    while True:
        try:
            # Coletar pergunta
            pergunta = input("\n👤 Digite sua pergunta sobre documentos (ou 'sair' para encerrar): ").strip()
            
            if pergunta.lower() in ['sair', 'exit', 'quit']:
                print("👋 Encerrando sistema...")
                break
            
            if not pergunta:
                continue
            
            estado.pergunta_usuario = pergunta
            print(f"[Pergunta] Processando: {pergunta}")
            
            # Buscar documentos
            print("[RAG] Buscando documentos relevantes...")
            try:
                from rag import DocumentProcessor
                processor = DocumentProcessor()
                
                docs = processor.buscar_documentos_similares(pergunta, max_results=5)
                
                if not docs:
                    print("❌ Nenhum documento relevante encontrado.")
                    continue
                
                # Mostrar documentos encontrados
                print(f"\n🔍 Encontrei {len(docs)} documentos relevantes:")
                print("=" * 80)
                
                for i, doc in enumerate(docs, 1):
                    arquivo = doc.metadata.get("arquivo", "Desconhecido")
                    chunk_id = doc.metadata.get("chunk_id", 0)
                    tipo = doc.metadata.get("tipo_arquivo", "desconhecido")
                    resumo = doc.page_content[:300].replace("\n", " ").strip()
                    
                    print(f"{i}. 📄 {arquivo}")
                    print(f"   📍 Chunk {chunk_id} | Tipo: {tipo}")
                    print(f"   📝 {resumo}...")
                    print()
                
                # Escolher documento
                if len(docs) > 1:
                    while True:
                        try:
                            escolha = input(f"🎯 Escolha um documento (1-{len(docs)}) ou 'auto' para o mais relevante: ").strip()
                            
                            if escolha.lower() == 'auto':
                                doc_escolhido = docs[0]
                                print(f"✅ Documento escolhido automaticamente: {doc_escolhido.metadata.get('arquivo', 'Desconhecido')}")
                                break
                            
                            if escolha.isdigit() and 1 <= int(escolha) <= len(docs):
                                idx = int(escolha) - 1
                                doc_escolhido = docs[idx]
                                print(f"✅ Documento escolhido: {doc_escolhido.metadata.get('arquivo', 'Desconhecido')}")
                                break
                            else:
                                print("❌ Opção inválida, tente novamente.")
                        except KeyboardInterrupt:
                            print("\n👋 Operação cancelada")
                            doc_escolhido = docs[0]
                            break
                else:
                    doc_escolhido = docs[0]
                    print(f"✅ Documento único encontrado: {doc_escolhido.metadata.get('arquivo', 'Desconhecido')}")
                
                # Executar consulta RAG
                print(f"\n[RAG] Executando consulta no documento escolhido...")
                resposta = processor.executar_rag(pergunta, max_results=1, auto_clarify=False)
                
                # Apresentar resultado
                print("\n" + "=" * 80)
                print("📋 RESULTADO DA CONSULTA")
                print("=" * 80)
                print(f"🔍 Pergunta: {pergunta}")
                print(f"📄 Documento: {doc_escolhido.metadata.get('arquivo', 'Desconhecido')}")
                print(f"📍 Chunk: {doc_escolhido.metadata.get('chunk_id', 0)}")
                print("-" * 80)
                print(resposta)
                print("=" * 80)
                
            except Exception as e:
                print(f"❌ Erro ao processar consulta: {str(e)}")
            
            # Perguntar se quer continuar
            continuar = input("\n🔄 Fazer nova consulta? (s/n): ").strip().lower()
            if continuar not in ['s', 'sim', 'y', 'yes']:
                print("👋 Encerrando sistema...")
                break
                
        except KeyboardInterrupt:
            print("\n\n👋 Sistema encerrado pelo usuário")
            break
        except Exception as e:
            print(f"\n❌ Erro inesperado: {str(e)}")

if __name__ == "__main__":
    main()
