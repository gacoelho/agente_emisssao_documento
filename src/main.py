from langgraph import graph, node
from dataclasses import dataclass, field

@dataclass
class EstadoFluxo:
    indice_existe: bool = False
    tipo_documento: str = ""
    docs_extraidos: list = field(default_factory=list)
    resposta: str = ""

def verificar_indice_node(estado: EstadoFluxo):
    import os
    INDEX_PATH = "/home/gacoelho/Documents/agente_emisssao_documento/doc"
    index_faiss = os.path.join(INDEX_PATH, "index.faiss")
    index_pkl = os.path.join(INDEX_PATH, "index.pkl")

    estado.indice_existe = os.path.exists(index_faiss) and os.path.exists(index_pkl)
    print(f"[Verificar Índice] Índice existe? {estado.indice_existe}")
    return estado.indice_existe

def perguntar_tipo_documento_node(estado: EstadoFluxo):
    tipos = ["Declaração de Estágio", "Declaração de Vínculo Empregatício", "Outro"]
    print("Qual tipo de documento você quer?")
    for i, t in enumerate(tipos, 1):
        print(f"{i}. {t}")
    escolha = input("Digite o número da opção: ").strip()
    try:
        idx = int(escolha) - 1
        estado.tipo_documento = tipos[idx]
    except Exception:
        estado.tipo_documento = "Outro"
    print(f"[Tipo Documento] Usuário escolheu: {estado.tipo_documento}")
    return estado.tipo_documento

def criar_indice_node(estado: EstadoFluxo):
    print("[Criar Índice] Criando índice FAISS com documentos...")
    # Aqui você chamaria sua função criar_indice_faiss(embeddings)
    # por simplicidade, só simulando:
    estado.indice_existe = True
    return True

def pergunta_usuario_node(estado: EstadoFluxo):
    pergunta = input("Você: ")
    # Aqui você chamaria executar_rag(pergunta) ou o pipeline real
    # Vou simular a resposta
    estado.resposta = f"Resposta simulada para pergunta '{pergunta}' considerando tipo doc '{estado.tipo_documento}'."
    print("\nBot:", estado.resposta)
    return estado.resposta

def main():
    estado = EstadoFluxo()
    graph = graph()

    n_verificar = node("verificar_indice", func=lambda: verificar_indice_node(estado))
    n_perguntar_tipo = node("perguntar_tipo_doc", func=lambda: perguntar_tipo_documento_node(estado))
    n_criar_indice = node("criar_indice", func=lambda: criar_indice_node(estado))
    n_pergunta_usuario = node("pergunta_usuario", func=lambda: pergunta_usuario_node(estado))

    graph.add_node(n_verificar)
    graph.add_node(n_perguntar_tipo)
    graph.add_node(n_criar_indice)
    graph.add_node(n_pergunta_usuario)

    # Fluxo com ramificações baseado no estado
    graph.add_edge(n_verificar, n_perguntar_tipo, condition=lambda res: not res)
    graph.add_edge(n_verificar, n_pergunta_usuario, condition=lambda res: res)
    graph.add_edge(n_perguntar_tipo, n_criar_indice)
    graph.add_edge(n_criar_indice, n_pergunta_usuario)

    graph.run()

if __name__ == "__main__":
    main()
