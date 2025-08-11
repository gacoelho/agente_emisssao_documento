import os
import dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from ocr import tool_ocr
from typing import List, Dict, Any, Optional
import logging

dotenv.load_dotenv()

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INDEX_PATH = "/home/gacoelho/Documents/agente_emisssao_documento/doc"

class DocumentProcessor:
    """Classe para processar e gerenciar documentos com desambiguação inteligente"""
    
    def __init__(self):
        self.embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            model=os.getenv("EMBEDDINGS_MODEL_NAME", "text-embedding-ada-002")
        )
        
        self.llm = AzureChatOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_KEY"),
            temperature=0.3
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def extrair_documentos_por_ocr(self, pasta_docs: str) -> List[Document]:
        """Extrai texto de documentos na pasta via OCR e retorna lista de Documentos"""
        documentos = []
        
        if not os.path.exists(pasta_docs):
            logger.warning(f"Pasta {pasta_docs} não encontrada")
            return documentos
        
        # Verificar se há documentos processáveis
        arquivos_processaveis = []
        for arquivo in os.listdir(pasta_docs):
            caminho = os.path.join(pasta_docs, arquivo)
            if os.path.isfile(caminho) and arquivo.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg', '.tiff')):
                arquivos_processaveis.append(arquivo)
        
        if not arquivos_processaveis:
            logger.warning(f"Nenhum arquivo processável encontrado em {pasta_docs}")
            return documentos
        
        logger.info(f"Processando {len(arquivos_processaveis)} arquivos...")
        
        for arquivo in arquivos_processaveis:
            caminho = os.path.join(pasta_docs, arquivo)
            try:
                logger.info(f"Processando arquivo: {arquivo}")
                resultado = tool_ocr(caminho)
                texto = resultado.get("texto_extraido", "").strip()
                
                if texto:
                    # Dividir o texto em chunks menores
                    chunks = self.text_splitter.split_text(texto)
                    
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={
                                "arquivo": arquivo,
                                "chunk_id": i,
                                "num_paginas": resultado.get("num_paginas", 0),
                                "tipo_arquivo": arquivo.split('.')[-1].lower(),
                                "tamanho_chunk": len(chunk)
                            }
                        )
                        documentos.append(doc)
                    
                    logger.info(f"Arquivo {arquivo} processado: {len(chunks)} chunks criados")
                else:
                    logger.warning(f"Nenhum texto extraído do arquivo {arquivo}")
                    
            except Exception as e:
                logger.error(f"Erro ao processar arquivo {arquivo}: {str(e)}")
        
        return documentos
    
    def criar_indice_faiss(self) -> bool:
        """Cria índice FAISS a partir dos documentos extraídos via OCR"""
        logger.info("Criando índice FAISS via OCR dos documentos na pasta...")
        
        try:
            docs = self.extrair_documentos_por_ocr(INDEX_PATH)
            
            if not docs:
                logger.warning("Nenhum documento extraído via OCR para criar índice.")
                return False
            
            logger.info(f"{len(docs)} chunks de documentos processados")
            
            # Criar o índice FAISS
            vectorstore = FAISS.from_documents(docs, self.embeddings)
            
            # Salvar o índice
            os.makedirs(INDEX_PATH, exist_ok=True)
            vectorstore.save_local(INDEX_PATH)
            
            logger.info(f"Índice FAISS criado e salvo em {INDEX_PATH}")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao criar índice FAISS: {str(e)}")
            return False
    
    def carregar_indice(self) -> Optional[FAISS]:
        """Carrega o índice FAISS existente ou cria um novo se necessário"""
        try:
            index_faiss = os.path.join(INDEX_PATH, "index.faiss")
            index_pkl = os.path.join(INDEX_PATH, "index.pkl")
            
            if not (os.path.exists(index_faiss) and os.path.exists(index_pkl)):
                logger.info("Índice não encontrado, criando novo...")
                if not self.criar_indice_faiss():
                    return None
            
            vectorstore = FAISS.load_local(
                INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info("Índice FAISS carregado com sucesso")
            return vectorstore
            
        except Exception as e:
            logger.error(f"Erro ao carregar índice: {str(e)}")
            return None
    
    def escolher_documento_opcoes(self, opcoes: List[Document], pergunta: str) -> Document:
        """Permite ao usuário escolher entre múltiplos documentos relevantes com contexto"""
        print(f"\n🔍 Encontrei {len(opcoes)} documentos relevantes para: '{pergunta}'")
        print("=" * 80)
        
        for i, doc in enumerate(opcoes, 1):
            arquivo = doc.metadata.get("arquivo", "Desconhecido")
            chunk_id = doc.metadata.get("chunk_id", 0)
            tipo_arquivo = doc.metadata.get("tipo_arquivo", "desconhecido")
            tamanho = doc.metadata.get("tamanho_chunk", 0)
            
            # Mostrar contexto relevante
            resumo = doc.page_content[:200].replace("\n", " ").strip()
            
            print(f"{i}. 📄 {arquivo}")
            print(f"   📍 Chunk {chunk_id} | Tipo: {tipo_arquivo} | Tamanho: {tamanho} chars")
            print(f"   📝 {resumo}...")
            print()
        
        while True:
            try:
                escolha = input(f"🎯 Escolha um documento (1-{len(opcoes)}) ou 'auto' para seleção automática: ").strip()
                
                if escolha.lower() == 'auto':
                    # Seleção automática baseada na relevância (primeiro resultado)
                    logger.info("Seleção automática ativada")
                    return opcoes[0]
                
                if escolha.isdigit() and 1 <= int(escolha) <= len(opcoes):
                    return opcoes[int(escolha) - 1]
                else:
                    print("❌ Opção inválida, tente novamente.")
            except KeyboardInterrupt:
                print("\n👋 Operação cancelada pelo usuário")
                return opcoes[0]  # Retorna a primeira opção como padrão
    
    def executar_rag(self, pergunta: str, max_results: int = 5, auto_clarify: bool = True) -> str:
        """Executa o pipeline RAG com Azure OpenAI e FAISS com desambiguação inteligente"""
        try:
            # Carregar ou criar índice
            vectorstore = self.carregar_indice()
            if not vectorstore:
                return "❌ Erro: Não foi possível carregar ou criar o índice de documentos."
            
            # Configurar retriever sem score threshold para debug
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": max_results}
            )
            
            # Buscar documentos relevantes usando o método correto
            try:
                docs = retriever.invoke(pergunta)
            except AttributeError:
                # Fallback para versões mais antigas
                docs = retriever.get_relevant_documents(pergunta)
            
            if not docs:
                return "❌ Nenhum documento relevante encontrado para sua pergunta. Tente reformular ou verificar se há documentos na pasta."
            
            # Se tiver mais de uma opção relevante e auto_clarify estiver ativo
            if len(docs) > 1 and auto_clarify:
                print(f"\n📚 Encontrei {len(docs)} documentos relevantes")
                doc_escolhido = self.escolher_documento_opcoes(docs, pergunta)
            else:
                doc_escolhido = docs[0]
            
            # Preparar contexto para o LLM
            contexto = doc_escolhido.page_content
            arquivo = doc_escolhido.metadata.get("arquivo", "Desconhecido")
            
            # Prompt melhorado para o LLM com contexto da pergunta
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""Você é um assistente especializado em documentos que ajuda usuários a entender o conteúdo de documentos.

📋 **Conteúdo do documento:**
{context}

❓ **Pergunta do usuário:** {question}

💡 **Instruções:**
Baseado APENAS no contexto fornecido, responda à pergunta do usuário de forma clara e precisa.
- Se a informação estiver disponível no contexto, forneça uma resposta completa
- Se a informação NÃO estiver disponível no contexto, indique claramente isso
- Seja profissional, objetivo e direto ao ponto
- Use o contexto específico do documento para fundamentar sua resposta

🔍 **Resposta baseada no documento:**"""
            )
            
            # Executar a consulta
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            resposta = qa_chain.invoke({"query": pergunta})
            
            return f"📄 **Documento consultado:** {arquivo}\n\n{resposta['result']}"
            
        except Exception as e:
            error_msg = f"❌ Erro ao executar consulta RAG: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def buscar_documentos_similares(self, texto: str, max_results: int = 3) -> List[Document]:
        """Busca documentos similares a um texto específico"""
        try:
            vectorstore = self.carregar_indice()
            if not vectorstore:
                return []
            
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": max_results}
            )
            
            # Usar o método correto baseado na versão
            try:
                return retriever.invoke(texto)
            except AttributeError:
                # Fallback para versões mais antigas
                return retriever.get_relevant_documents(texto)
            
        except Exception as e:
            logger.error(f"Erro ao buscar documentos similares: {str(e)}")
            return []
    
    def obter_estatisticas_indice(self) -> Dict[str, Any]:
        """Retorna estatísticas detalhadas sobre o índice atual"""
        try:
            vectorstore = self.carregar_indice()
            if not vectorstore:
                return {"status": "erro", "mensagem": "Índice não disponível"}
            
            # Contar documentos
            num_docs = len(vectorstore.docstore._dict)
            
            # Obter tipos de arquivos únicos
            tipos_arquivo = set()
            arquivos = set()
            total_chunks = 0
            
            for doc_id, doc in vectorstore.docstore._dict.items():
                if hasattr(doc, 'metadata'):
                    tipos_arquivo.add(doc.metadata.get('tipo_arquivo', 'desconhecido'))
                    arquivos.add(doc.metadata.get('arquivo', 'desconhecido'))
                    total_chunks += 1
            
            return {
                "status": "ativo",
                "total_documentos": total_chunks,
                "arquivos_unicos": len(arquivos),
                "tipos_arquivo": list(tipos_arquivo),
                "caminho_indice": INDEX_PATH,
                "ultima_atualizacao": "Agora"
            }
            
        except Exception as e:
            return {"status": "erro", "mensagem": str(e)}
    
    def forcar_recriacao_indice(self) -> bool:
        """Força a recriação do índice FAISS"""
        try:
            # Remover arquivos existentes
            index_faiss = os.path.join(INDEX_PATH, "index.faiss")
            index_pkl = os.path.join(INDEX_PATH, "index.pkl")
            
            if os.path.exists(index_faiss):
                os.remove(index_faiss)
            if os.path.exists(index_pkl):
                os.remove(index_pkl)
            
            logger.info("Arquivos de índice removidos, recriando...")
            return self.criar_indice_faiss()
            
        except Exception as e:
            logger.error(f"Erro ao forçar recriação do índice: {str(e)}")
            return False

# Funções de compatibilidade para uso externo
def extrair_documentos_por_ocr(pasta_docs: str) -> list[Document]:
    """Função de compatibilidade para uso externo"""
    processor = DocumentProcessor()
    return processor.extrair_documentos_por_ocr(pasta_docs)

def escolher_documento_opcoes(opcoes: list[Document]) -> Document:
    """Função de compatibilidade para uso externo"""
    processor = DocumentProcessor()
    return processor.escolher_documento_opcoes(opcoes, "Consulta")

def criar_indice_faiss(embeddings):
    """Função de compatibilidade para uso externo"""
    processor = DocumentProcessor()
    return processor.criar_indice_faiss()

def executar_rag(pergunta: str) -> str:
    """Função de compatibilidade para uso externo"""
    processor = DocumentProcessor()
    return processor.executar_rag(pergunta)

if __name__ == "__main__":
    # Teste do sistema
    print("🧪 Teste do Sistema RAG")
    print("=" * 40)
    
    processor = DocumentProcessor()
    
    # Verificar estatísticas
    stats = processor.obter_estatisticas_indice()
    print(f"📊 Estatísticas do índice: {stats}")
    
    # Teste de consulta
    if stats["status"] == "ativo":
        pergunta = input("\n🔍 Digite uma pergunta para testar: ")
        if pergunta.strip():
            resposta = processor.executar_rag(pergunta)
            print(f"\n🤖 Resposta: {resposta}")
    else:
        print(f"❌ Erro no índice: {stats['mensagem']}")
        print("🔄 Tentando criar índice...")
        if processor.criar_indice_faiss():
            print("✅ Índice criado com sucesso!")
        else:
            print("❌ Falha ao criar índice")
