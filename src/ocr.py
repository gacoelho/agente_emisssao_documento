from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
from dotenv import load_dotenv
import time

load_dotenv()

def tool_ocr(doc_path: str) -> dict:
    start = time.time()

    try:
        doc_client = DocumentAnalysisClient(
            endpoint=os.getenv("AZURE_DOC_INT"),
            credential=AzureKeyCredential(os.getenv("AZ_KEY"))
        )

        with open(doc_path, "rb") as f:
            poller = doc_client.begin_analyze_document("prebuilt-document", document=f)
            result = poller.result()

        conteudo_extraido = "\n".join([p.content for p in result.pages[0].lines])
        tempo_total = time.time() - start

        print(f"Tempo de execução do OCR: %.2f segundos", tempo_total)
        return {
            "texto_extraido": conteudo_extraido,
            "num_paginas": len(result.pages),
            "tempo_ocr": tempo_total
        }

    except Exception as e:
        print(f"[OCR TOOL] Falha ao executar OCR: {e}")
        return {"texto_extraido": "", "tempo_ocr": 0, "num_paginas": 0}