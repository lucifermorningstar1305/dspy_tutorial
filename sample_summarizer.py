import dspy
import json
import chromadb

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from dspy.retrieve import chromadb_rm


if __name__ == "__main__":

    pipeline = PdfPipelineOptions()
    pipeline.do_ocr = False
    pipeline.do_table_structure = True
    pipeline.table_structure_options.do_cell_matching = True

    source = "https://arxiv.org/pdf/1810.04805"
    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline, backend=PyPdfiumDocumentBackend
            )
        },
    )

    result = converter.convert(source).document.export_to_dict()
    with open("./test.json", "w") as fp:
        json.dump(result, indent=4, fp=fp)

    title = ""
    docs = list()
    metadata = list()
    ids = list()
    texts = result["texts"]

    for i in range(len(texts)):
        if texts[i]["label"] == "section_header":
            title = texts[i]["text"]
            break

    for i in range(len(texts)):
        if (
            texts[i]["text"].lower() in ["abstract", "summary"]
            or len(texts[i]["text"].lower().split()) >= 128
        ):
            break

    texts = texts[i:]
    header = ""

    for i in range(len(texts)):
        if texts[i]["label"].lower() == "section_header":
            header = texts[i]["text"]

        if texts[i]["label"].lower() in ["text", "list_item", "formula"]:
            docs.append(texts[i]["text"])
            metadata.append(
                {
                    "section": header,
                    "title": title,
                    "page_no": texts[i]["prov"][0]["page_no"],
                }
            )
            ids.append(f"{i:03d}")

    client = chromadb.PersistentClient("./db")
    collection = client.get_or_create_collection("sample_collection")
    if not collection.count():
        collection.add(ids=ids, documents=docs, metadatas=metadata)

    lm = dspy.LM(model="ollama_chat/llama3.1", temperature=0.2)
    dspy.configure(lm=lm)

    class RAG(dspy.Module):
        def __init__(self):
            self.retriever = chromadb_rm.ChromadbRM(
                collection_name="sample_collection", persist_directory="./db"
            )

            self.respond = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question):
            ref_docs = self.retriever(question)
            contexts = [doc["long_text"] for doc in ref_docs]
            return self.respond(context=contexts, question=question)

    rag = RAG()
    print(rag(question="How is the BERT model different from GPT?").answer)
    print(rag(question="Can you summarize this document?").answer)
