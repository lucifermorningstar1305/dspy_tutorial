from typing import List
import dspy
import json
import chromadb

from docling.datamodel.base_models import InputFormat
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

from dspy.retrieve import chromadb_rm


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


class ChunkSummarizer(dspy.Module):
    def __init__(self):
        self.summarizer = dspy.ChainOfThought("text -> summary")

    def forward(self, text: str) -> str:
        return self.summarizer(text=text).summary


class MapReduceSummarizer(dspy.Module):
    def __init__(self):
        self.chunk_summarizer = ChunkSummarizer()
        self.final_summarizer = dspy.ChainOfThought("summary -> final_summary")

    def forward(self, chunks: List[str]) -> str:
        chunk_summaries = [self.chunk_summarizer(chunk) for chunk in chunks]
        combined_summaries = " ".join(chunk_summaries)
        return self.final_summarizer(summary=combined_summaries).final_summary


def get_chunks(document: str, chunk_size: int = 1000) -> List[str]:
    """Returns chunks from the documents"""
    return [document[i : i + chunk_size] for i in range(0, len(document), chunk_size)]


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

    rag = RAG()
    print(rag(question="How is the BERT model different from GPT?").answer)
    print(rag(question="Can you summarize this document?").answer)

    summarizer = MapReduceSummarizer()
    chunks = get_chunks(" ".join(docs), chunk_size=5000)
    final_summary = summarizer(chunks)
    print(final_summary)
