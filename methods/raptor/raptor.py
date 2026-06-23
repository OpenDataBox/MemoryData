import numpy as np
import pandas as pd
from typing import List, Dict, Any

try:
    from langchain_classic.chains.llm import LLMChain
except ImportError:
    from langchain.chains.llm import LLMChain

from sklearn.mixture import GaussianMixture
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS

try:
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    from langchain.prompts import ChatPromptTemplate

try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever

try:
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    from langchain.retrievers.document_compressors import LLMChainExtractor

try:
    from langchain_core.messages import AIMessage
except ImportError:
    from langchain.schema import AIMessage

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.docstore.document import Document

import matplotlib.pyplot as plt
import logging
import os
import sys
import tiktoken
from dotenv import load_dotenv
import time
import re

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path

# Load environment variables from a .env file
load_dotenv()

from utils.provider_utils import (
    normalize_provider as _normalize_provider,
    resolve_env_value as _resolve_env_value,
    resolve_base_url as _resolve_base_url,
    use_azure_openai as _use_azure_openai,
    create_chat_llm as _create_chat_llm,
    create_embedding_model as _create_embedding_model,
)


# Helper functions

_RAPTOR_ENCODING = tiktoken.get_encoding("cl100k_base")
_SUMMARY_TOKEN_BUDGET = 24000
_ANSWER_CONTEXT_TOKEN_BUDGET = 24000


def _truncate_to_token_budget(text: str, token_budget: int) -> str:
    if not text or token_budget <= 0:
        return ""
    tokens = _RAPTOR_ENCODING.encode(text, disallowed_special=())
    if len(tokens) <= token_budget:
        return text
    return _RAPTOR_ENCODING.decode(tokens[:token_budget])

def extract_text(item):
    """Extract text content from either a string or an AIMessage object."""
    if isinstance(item, AIMessage):
        return item.content
    return item


def embed_texts(texts: List[str], embedding_model) -> List[List[float]]:
    """Embed texts using the configured embedding model."""
    logging.info(f"Embedding {len(texts)} texts")
    return embedding_model.embed_documents([extract_text(text) for text in texts])


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 10) -> np.ndarray:
    """Perform clustering on embeddings using Gaussian Mixture Model."""
    logging.info(f"Performing clustering with {n_clusters} clusters")
    gm = GaussianMixture(n_components=n_clusters, random_state=42)
    return gm.fit_predict(embeddings)


def summarize_texts(texts: List[str], llm) -> str:
    """Summarize a list of texts using the configured chat model."""
    logging.info(f"Summarizing {len(texts)} texts")
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text concisely:\n\n{text}"
    )
    chain = prompt | llm
    summary_source = "\n\n".join([extract_text(text) for text in texts])
    summary_source = _truncate_to_token_budget(summary_source, _SUMMARY_TOKEN_BUDGET)
    input_data = {"text": summary_source}
    return chain.invoke(input_data)


def visualize_clusters(embeddings: np.ndarray, labels: np.ndarray, level: int):
    """Visualize clusters using PCA."""
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title(f'Cluster Visualization - Level {level}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.show()


def build_vectorstore(tree_results: Dict[int, pd.DataFrame], embedding_model) -> FAISS:
    """Build a FAISS vectorstore from all texts in the RAPTOR tree."""
    all_texts = []
    all_embeddings = []
    all_metadatas = []

    for level, df in tree_results.items():
        all_texts.extend([str(text) for text in df['text'].tolist()])
        all_embeddings.extend([embedding.tolist() if isinstance(embedding, np.ndarray) else embedding for embedding in
                               df['embedding'].tolist()])
        all_metadatas.extend(df['metadata'].tolist())

    logging.info(f"Building vectorstore with {len(all_texts)} texts")
    documents = [Document(page_content=str(text), metadata=metadata)
                 for text, metadata in zip(all_texts, all_metadatas)]
    return FAISS.from_documents(documents, embedding_model)


def create_retriever(vectorstore: FAISS, llm) -> ContextualCompressionRetriever:
    """Create a retriever with contextual compression."""
    logging.info("Creating contextual compression retriever")
    base_retriever = vectorstore.as_retriever()

    prompt = ChatPromptTemplate.from_template(
        "Given the following context and question, extract only the relevant information for answering the question:\n\n"
        "Context: {context}\n"
        "Question: {question}\n\n"
        "Relevant Information:"
    )

    extractor = LLMChainExtractor.from_llm(llm, prompt=prompt)
    return ContextualCompressionRetriever(
        base_compressor=extractor,
        base_retriever=base_retriever
    )


class RAPTORMethod:
    def __init__(self, texts: List[str], max_levels: int = 3, model="gpt-4o-mini", temperature: float = 0.0,
                 provider="openai", base_url=None, base_url_env=None, api_key_env=None,
                 azure_endpoint=None, azure_api_version=None, embedding_model=None,
                 embedding_provider=None, embedding_base_url=None, embedding_base_url_env=None,
                 embedding_api_key_env=None, embedding_azure_endpoint=None,
                 embedding_azure_api_version=None):
        self.start_time = time.time()
        self.texts = texts
        self.max_levels = max_levels
        self.embeddings = _create_embedding_model(
            model=embedding_model,
            provider=embedding_provider,
            base_url=embedding_base_url,
            base_url_env=embedding_base_url_env,
            api_key_env=embedding_api_key_env,
            azure_endpoint=embedding_azure_endpoint,
            azure_api_version=embedding_azure_api_version,
        )
        self.llm = _create_chat_llm(
            model=model,
            temperature=temperature,
            provider=provider,
            base_url=base_url,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )
        self.tree_results = self.build_raptor_tree()

    def build_raptor_tree(self) -> Dict[int, pd.DataFrame]:
        """Build the RAPTOR tree structure with level metadata and parent-child relationships."""
        results = {}
        current_texts = [extract_text(text) for text in self.texts]
        current_metadata = [{"level": 0, "origin": "original", "parent_id": None} for _ in self.texts]

        for level in range(1, self.max_levels + 1):
            logging.info(f"Processing level {level}")

            embeddings = embed_texts(current_texts, self.embeddings)
            n_clusters = min(10, len(current_texts) // 2)
            cluster_labels = perform_clustering(np.array(embeddings), n_clusters)

            df = pd.DataFrame({
                'text': current_texts,
                'embedding': embeddings,
                'cluster': cluster_labels,
                'metadata': current_metadata
            })

            results[level - 1] = df

            summaries = []
            new_metadata = []
            for cluster in df['cluster'].unique():
                cluster_docs = df[df['cluster'] == cluster]
                cluster_texts = cluster_docs['text'].tolist()
                cluster_metadata = cluster_docs['metadata'].tolist()
                summary = summarize_texts(cluster_texts, self.llm)
                summaries.append(summary)
                new_metadata.append({
                    "level": level,
                    "origin": f"summary_of_cluster_{cluster}_level_{level - 1}",
                    "child_ids": [meta.get('id') for meta in cluster_metadata],
                    "id": f"summary_{level}_{cluster}"
                })

            current_texts = summaries
            current_metadata = new_metadata

            if len(current_texts) <= 1:
                results[level] = pd.DataFrame({
                    'text': current_texts,
                    'embedding': embed_texts(current_texts, self.embeddings),
                    'cluster': [0],
                    'metadata': current_metadata
                })
                logging.info(f"Stopping at level {level} as we have only one summary")
                break

        return results

    def run(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Run the RAPTOR query pipeline."""
        vectorstore = build_vectorstore(self.tree_results, self.embeddings)
        retriever = create_retriever(vectorstore, self.llm)

        logging.info(f"Processing query: {query}")
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query = ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query = ''.join(match.groups())
            else:
                retrieval_query = query
        print(f"Retrieve query: {retrieval_query}")
        if hasattr(retriever, "invoke"):
            relevant_docs = retriever.invoke(retrieval_query)
        else:
            relevant_docs = retriever.get_relevant_documents(retrieval_query)

        doc_details = [{"content": doc.page_content, "metadata": doc.metadata} for doc in relevant_docs]

        context_parts = []
        remaining_budget = _ANSWER_CONTEXT_TOKEN_BUDGET
        for doc in relevant_docs:
            clipped_content = _truncate_to_token_budget(doc.page_content, remaining_budget)
            if not clipped_content:
                break
            context_parts.append(clipped_content)
            remaining_budget -= len(_RAPTOR_ENCODING.encode(clipped_content, disallowed_special=()))
            if remaining_budget <= 0:
                break

        context = "\n\n".join(context_parts)
        prompt = ChatPromptTemplate.from_template(
            "Given the following context, please answer the question:\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n\n"
        )
        memory_construction_time = time.time() - self.start_time
        chain = LLMChain(llm=self.llm, prompt=prompt)
        answer = chain.run(context=context, question=query)
        query_time_len = time.time() - self.start_time - memory_construction_time

        return {
            "query": query,
            "retrieved_documents": doc_details,
            "context_used": context,
            "answer": answer,
            "model_used": getattr(self.llm, 'model_name', getattr(self.llm, 'model', 'unknown')),
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
        }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAPTORMethod")
    parser.add_argument("--path", type=str, default="../data/Understanding_Climate_Change.pdf",
                        help="Path to the PDF file to process.")
    parser.add_argument("--query", type=str, default="What is the greenhouse effect?",
                        help="Query to test the retriever (default: 'What is the main topic of the document?').")
    parser.add_argument('--max_levels', type=int, default=3, help="Max levels for RAPTOR tree")
    return parser.parse_args()


if __name__ == "__main__":
    pass
