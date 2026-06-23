import torch
import os
import httpx
from transformers import AutoTokenizer, AutoModel
import numpy as np
from typing import List, Dict, Union
from openai import OpenAI, AzureOpenAI
import torch.nn.functional as F
from benchmark.memoryagentbench.loader import format_chat
import time
import re

from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings

try:
    from langchain_core.embeddings import Embeddings
except ImportError:
    from langchain.embeddings.base import Embeddings

from langchain_community.vectorstores import FAISS

try:
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import Document

from tqdm import tqdm


from utils.provider_utils import (
    normalize_provider as _normalize_provider,
    resolve_env_value as _resolve_env_value,
    resolve_base_url as _resolve_base_url,
    use_azure_openai as _use_azure_openai,
)


def _should_disable_qwen3_thinking(model, disable_flag=True):
    """Return whether Qwen3 requests should disable thinking mode."""
    model_name = str(model or "").strip().lower()
    if "qwen3" not in model_name:
        return False
    return bool(disable_flag)


def _prepare_qwen3_request_kwargs(model, request_kwargs, disable_flag=True):
    """Inject Qwen3 compatibility knobs for OpenAI-compatible chat requests."""
    if not _should_disable_qwen3_thinking(model, disable_flag=disable_flag):
        return request_kwargs

    extra_body = dict(request_kwargs.get("extra_body") or {})
    extra_body.setdefault("enable_thinking", False)
    chat_template_kwargs = dict(extra_body.get("chat_template_kwargs") or {})
    chat_template_kwargs.setdefault("enable_thinking", False)
    extra_body["chat_template_kwargs"] = chat_template_kwargs
    request_kwargs["extra_body"] = extra_body
    return request_kwargs


def _create_embedding_model_client(model, provider="openai", base_url=None, base_url_env=None,
                                   api_key_env=None, azure_endpoint=None, azure_api_version="2024-02-01",
                                   azure_api_key=None, azure_deployment=None):
    provider = _normalize_provider(provider)
    if _use_azure_openai(provider, azure_endpoint=azure_endpoint, base_url=base_url, base_url_env=base_url_env):
        return AzureOpenAIEmbeddings(
            model=model,
            azure_deployment=azure_deployment or model,
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=azure_api_key or _resolve_env_value(api_key_env, ["AZURE_OPENAI_API_KEY"]),
            api_version=azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
        )

    resolved_base_url = _resolve_base_url(base_url, base_url_env)
    api_key = _resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
    if provider == "openai_compatible" and not resolved_base_url:
        raise RuntimeError("OpenAI-compatible embedding models require 'base_url' or 'base_url_env'.")

    kwargs = {"model": model}
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if api_key:
        kwargs["api_key"] = api_key
    if resolved_base_url:
        kwargs["http_client"] = httpx.Client(trust_env=False)
    return OpenAIEmbeddings(**kwargs)


def _create_llm_client(provider="openai", base_url=None, base_url_env=None, api_key_env=None,
                       azure_endpoint=None, azure_api_version="2024-02-01", azure_api_key=None):
    provider = _normalize_provider(provider)
    if _use_azure_openai(provider, azure_endpoint=azure_endpoint, base_url=base_url, base_url_env=base_url_env):
        return AzureOpenAI(
            azure_endpoint=azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT"),
            api_key=azure_api_key or _resolve_env_value(api_key_env, ["AZURE_OPENAI_API_KEY"]),
            api_version=azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION"),
        )

    resolved_base_url = _resolve_base_url(base_url, base_url_env)
    api_key = _resolve_env_value(api_key_env, ["OPENAI_API_KEY"])
    if provider == "openai_compatible" and not resolved_base_url:
        raise RuntimeError("OpenAI-compatible models require 'base_url' or 'base_url_env'.")

    kwargs = {}
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url
    if api_key:
        kwargs["api_key"] = api_key
    if resolved_base_url:
        kwargs["http_client"] = httpx.Client(trust_env=False)
    return OpenAI(**kwargs)

# Create a custom embedding class for Contriever
class ContrieverEmbeddings(Embeddings):
    def __init__(self, model_name="facebook/contriever"):
        assert "contriever" in model_name, "Model name must contain 'contriever'"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in tqdm(texts, desc="Embedding documents (Contriever)"):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            embedding = outputs.last_hidden_state[:, 0, :]
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings
    
    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        embedding = outputs.last_hidden_state[:, 0, :]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()


class Qwen3Embedding4BEmbeddings(Embeddings):
    def __init__(self, model_name="Qwen/Qwen3-Embedding-4B"):
        assert "Qwen3-Embedding-4B" in model_name or "Qwen/Qwen3-Embedding-4B" in model_name, "Model name must be Qwen/Qwen3-Embedding-4B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()
        
    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        # Batch encode for efficiency if needed; keep simple loop for parity with ContrieverEmbeddings
        for text in tqdm(texts, desc="Embedding documents (Qwen3-Embedding-4B)"):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings
        
    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()


class NVEmbedV2Embeddings(Embeddings):
    def __init__(self, model_name="nvidia/NV-Embed-v2"):
        assert "NV" in model_name or "nv" in model_name, "Model name should be an NV-Embed variant"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()

    def _mean_pooling(self, last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = (last_hidden_state * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in tqdm(texts, desc="Embedding documents (NV-Embed-v2)"):
            inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
            embedding = F.normalize(embedding, p=2, dim=1)
            embeddings.append(embedding.cpu().numpy()[0].tolist())
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        last_hidden = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
        embedding = self._mean_pooling(last_hidden, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()[0].tolist()

class  TextRetriever:
    def __init__(self,
                 embedding_model_name: str = "text-embedding-3-large",
                 sub_dataset=None,
                 provider: str = "openai",
                 base_url: str = None,
                 base_url_env: str = None,
                 api_key_env: str = None,
                 use_azure: bool = False,
                 azure_endpoint: str = None,
                 azure_api_key: str = None,
                 azure_api_version: str = "2024-02-01",
                 azure_embedding_deployment: str = None):
        provider = _normalize_provider(provider or ("azure_openai" if use_azure else "openai"))

        if use_azure and not azure_embedding_deployment:
            azure_embedding_deployment = embedding_model_name

        if use_azure and not azure_api_key:
            azure_api_key = os.environ.get("AZURE_OPENAI_API_KEY")

        if use_azure and not azure_endpoint:
            azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")

        if use_azure and not azure_api_version:
            azure_api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if use_azure and provider == "openai":
            provider = "azure_openai"

        if embedding_model_name == "facebook/contriever":
            self.embedding_model = ContrieverEmbeddings(model_name=embedding_model_name)
        elif embedding_model_name == "Qwen/Qwen3-Embedding-4B":
            self.embedding_model = Qwen3Embedding4BEmbeddings(model_name=embedding_model_name)
        elif embedding_model_name in ["nvidia/NV-Embed-v2", "nvidia/NV-Embed-v2-7B", "NV-Embed-v2-7B", "NV-Embed-v2"]:
            self.embedding_model = NVEmbedV2Embeddings(model_name=embedding_model_name)
        else:
            self.embedding_model = _create_embedding_model_client(
                model=embedding_model_name,
                provider=provider,
                base_url=base_url,
                base_url_env=base_url_env,
                api_key_env=api_key_env,
                azure_endpoint=azure_endpoint,
                azure_api_version=azure_api_version,
                azure_api_key=azure_api_key,
                azure_deployment=azure_embedding_deployment,
            )
        self.sub_dataset = sub_dataset
        self.vectorstore: FAISS = None
        self._current_documents = None
        
    def build_vectorstore(self, documents: List[str]):
        """Build and cache the vector store from documents"""
        # Convert strings to Document objects if needed
        if isinstance(documents[0], str):
            doc_objects = [Document(page_content=doc) for doc in documents]
        else:
            doc_objects = documents
            
        self.vectorstore = FAISS.from_documents(doc_objects, self.embedding_model)
        self._current_documents = documents
        
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve most relevant contexts for a query (auto-caches vectorstore)
        
        Args:
            query: The search query
            top_k: Number of documents to retrieve from vector store
        """
        initial_k = top_k
        
        # Perform similarity search to get initial results
        results = self.vectorstore.similarity_search(query, k=initial_k)
        retrieved_docs = [doc.page_content for doc in results]
        
        # Return results (truncated to top_k if needed)
        return retrieved_docs[:top_k]
    



class RAGSystem:
    def __init__(self,
                 retriever,
                 model,
                 temperature,
                 max_tokens,
                 provider: str = "openai",
                 base_url: str = None,
                 base_url_env: str = None,
                 api_key_env: str = None,
                 use_azure: bool = False,
                 azure_endpoint: str = None,
                 azure_api_key: str = None,
                 azure_api_version: str = "2024-02-01",
                 qwen3_disable_thinking: bool = True,
                 prompt_token_budget: int = None,
                 tokenizer=None):
        self.retriever = retriever
        provider = _normalize_provider(provider or ("azure_openai" if use_azure else "openai"))
        if use_azure and provider == "openai":
            provider = "azure_openai"

        self.llm = _create_llm_client(
            provider=provider,
            base_url=base_url,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
            azure_api_key=azure_api_key,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.qwen3_disable_thinking = qwen3_disable_thinking
        self.prompt_token_budget = prompt_token_budget
        self.tokenizer = tokenizer

    def _encode_tokens(self, text: str):
        if not self.tokenizer:
            return list(text or "")
        try:
            return self.tokenizer.encode(text, disallowed_special=())
        except TypeError:
            return self.tokenizer.encode(text)

    def _decode_tokens(self, tokens):
        if self.tokenizer and hasattr(self.tokenizer, "decode"):
            return self.tokenizer.decode(tokens)
        return "".join(tokens)

    def _fit_contexts_to_prompt_budget(
        self,
        contexts: List[str],
        query: str,
        system_message: str = "",
        reserved_tokens: int = 4096,
    ) -> List[str]:
        if not contexts or not self.prompt_token_budget:
            return contexts

        system_tokens = len(self._encode_tokens(system_message or ""))
        budget = max(256, self.prompt_token_budget - system_tokens - reserved_tokens)
        used_tokens = len(self._encode_tokens(query))
        fitted_contexts = []

        for index, text in enumerate(contexts, start=1):
            prefix = f"Memory {index}:\n"
            prefix_tokens = len(self._encode_tokens(prefix))
            text_tokens = self._encode_tokens(text)
            remaining = budget - used_tokens - prefix_tokens

            if remaining <= 0:
                break

            if len(text_tokens) > remaining:
                truncated_text = self._decode_tokens(text_tokens[:remaining]).strip()
                if truncated_text:
                    fitted_contexts.append(truncated_text)
                break

            fitted_contexts.append(text)
            used_tokens += prefix_tokens + len(text_tokens)

        if fitted_contexts:
            return fitted_contexts

        fallback_prefix_tokens = len(self._encode_tokens("Memory 1:\n"))
        fallback_budget = max(0, budget - used_tokens - fallback_prefix_tokens)
        if fallback_budget <= 0:
            return []

        fallback_tokens = self._encode_tokens(contexts[0])[:fallback_budget]
        fallback_text = self._decode_tokens(fallback_tokens).strip()
        return [fallback_text] if fallback_text else []

    def answer_query(self, query: str, top_k: int, system_message: str) -> Dict[str, Union[str, float]]:
        """Retrieve relevant information and generate an answer"""
        # Retrieve relevant passages
        start_time = time.time()
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query =  ''.join(match.groups())
            else:
                retrieval_query = query
        print(f"Retrieve query: {retrieval_query}")
        retrieved_contexts = self.retriever.retrieve(retrieval_query, top_k)
        memory_construction_time = time.time() - start_time
        reserve_attempts = [
            max(self.max_tokens + 2048, 4096),
            6144,
            8192,
            12288,
        ]
        reserve_attempts = list(dict.fromkeys(reserve_attempts))

        last_exception = None
        for reserved_tokens in reserve_attempts:
            fitted_contexts = self._fit_contexts_to_prompt_budget(
                retrieved_contexts,
                query,
                system_message=system_message,
                reserved_tokens=reserved_tokens,
            )

            formatted_context = "\n\n".join(
                [f"Passage {i+1}:\n{text}" for i, text in enumerate(fitted_contexts)]
            )
            retrieval_memory_string = "\n".join(
                [f"Memory {i+1}:\n{text}" for i, text in enumerate(fitted_contexts)]
            )
            ask_llm_message = "\n".join(part for part in [retrieval_memory_string, query] if part)
            format_message = format_chat(message=ask_llm_message, system_message=system_message)

            request_kwargs = _prepare_qwen3_request_kwargs(
                self.model,
                {
                    "model": self.model,
                    "messages": format_message,
                    "temperature": self.temperature,
                    "max_tokens": self.max_tokens,
                },
                disable_flag=self.qwen3_disable_thinking,
            )
            try:
                response = self.llm.chat.completions.create(**request_kwargs)
                query_time_len = time.time() - start_time - memory_construction_time
                break
            except Exception as exc:
                if "maximum context length" not in str(exc).lower():
                    raise
                last_exception = exc
        else:
            raise last_exception

        return {
            "query": query,
            "context_used": formatted_context,
            "answer": response.choices[0].message.content,
            "memory_construction_time": memory_construction_time,
            "query_time_len": query_time_len,
        }
