import os
import sys
from dotenv import load_dotenv

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
import time
import re

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path since we work with notebooks

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


class RetrievalResponse(BaseModel):
    response: str = Field(..., title="Determines if retrieval is necessary", description="Output only 'Yes' or 'No'.")


class RelevanceResponse(BaseModel):
    response: str = Field(..., title="Determines if context is relevant",
                          description="Output only 'Relevant' or 'Irrelevant'.")


class GenerationResponse(BaseModel):
    response: str = Field(..., title="Generated response", description="The generated response.")


class SupportResponse(BaseModel):
    response: str = Field(..., title="Determines if response is supported",
                          description="Output 'Fully supported', 'Partially supported', or 'No support'.")


class UtilityResponse(BaseModel):
    response: int = Field(..., title="Utility rating", description="Rate the utility of the response from 1 to 5.")


retrieval_prompt = PromptTemplate(
    input_variables=["query"],
    template="Given the query '{query}', determine if retrieval is necessary. Output only 'Yes' or 'No'."
)

relevance_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', determine if the context is relevant. Output only 'Relevant' or 'Irrelevant'."
)

generation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="Given the query '{query}' and the context '{context}', generate a response."
)

support_prompt = PromptTemplate(
    input_variables=["response", "context"],
    template="Given the response '{response}' and the context '{context}', determine if the response is supported by the context. Output 'Fully supported', 'Partially supported', or 'No support'."
)

utility_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Given the query '{query}' and the response '{response}', rate the utility of the response from 1 to 5."
)


def replace_t_with_space(list_of_documents):
    """Replace all tab characters with spaces in each document."""
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')
    return list_of_documents


def encode_documents(documents, embedding_model, chunk_size=4096, chunk_overlap=200):
    """Encode documents into a FAISS vector store using the configured embedding model."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    texts = text_splitter.split_documents(documents)
    cleaned_texts = replace_t_with_space(texts)
    return FAISS.from_documents(cleaned_texts, embedding_model)


class SelfRAG:
    def __init__(self, documents, temperature=0.7, top_k=3, model="gpt-4o-mini",
                 provider="openai", base_url=None, base_url_env=None, api_key_env=None,
                 azure_endpoint=None, azure_api_version=None, embedding_model=None,
                 embedding_provider=None, embedding_base_url=None, embedding_base_url_env=None,
                 embedding_api_key_env=None, embedding_azure_endpoint=None,
                 embedding_azure_api_version=None):
        embedding_client = _create_embedding_model(
            model=embedding_model,
            provider=embedding_provider,
            base_url=embedding_base_url,
            base_url_env=embedding_base_url_env,
            api_key_env=embedding_api_key_env,
            azure_endpoint=embedding_azure_endpoint,
            azure_api_version=embedding_azure_api_version,
        )
        self.vectorstore = encode_documents(documents=documents, embedding_model=embedding_client)
        self.top_k = top_k
        self.llm = _create_chat_llm(
            model=model,
            temperature=temperature,
            max_tokens=1000,
            provider=provider,
            base_url=base_url,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )

        self.retrieval_chain = retrieval_prompt | self.llm.with_structured_output(RetrievalResponse)
        self.relevance_chain = relevance_prompt | self.llm.with_structured_output(RelevanceResponse)
        self.generation_chain = generation_prompt | self.llm.with_structured_output(GenerationResponse)
        self.support_chain = support_prompt | self.llm.with_structured_output(SupportResponse)
        self.utility_chain = utility_prompt | self.llm.with_structured_output(UtilityResponse)
        self.start_time = time.time()

    def run(self, query):
        print(f"\nProcessing query: {query}")
        print("Step 1: Determining if retrieval is necessary...")

        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query = ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query = ''.join(match.groups())
            else:
                retrieval_query = query
        print(f"\n\nRetrieval query: {retrieval_query}\n\n")

        input_data = {"query": retrieval_query}
        retrieval_decision = self.retrieval_chain.invoke(input_data).response.strip().lower()
        print(f"Retrieval decision: {retrieval_decision}")

        if retrieval_decision == 'yes':
            print("Step 2: Retrieving relevant documents...")
            docs = self.vectorstore.similarity_search(retrieval_query, k=self.top_k)
            contexts = [doc.page_content for doc in docs]
            print(f"Retrieved {len(contexts)} documents")

            print("Step 3: Evaluating relevance of retrieved documents...")
            relevant_contexts = []
            for i, context in enumerate(contexts):
                input_data = {"query": retrieval_query, "context": context}
                relevance = self.relevance_chain.invoke(input_data).response.strip().lower()
                print(f"Document {i + 1} relevance: {relevance}")
                if relevance == 'relevant':
                    relevant_contexts.append(context)

            print(f"Number of relevant contexts: {len(relevant_contexts)}")

            memory_construction_time = time.time() - self.start_time
            if not relevant_contexts:
                print("No relevant contexts found. Generating without retrieval...")
                input_data = {"query": query, "context": "No relevant context found."}
                return self.generation_chain.invoke(input_data).response, "No relevant context found.", memory_construction_time, (time.time() - self.start_time - memory_construction_time)

            print("Step 4: Generating responses using relevant contexts...")
            responses = []
            for i, context in enumerate(relevant_contexts):
                print(f"Generating response for context {i + 1}...")
                input_data = {"query": query, "context": context}
                response = self.generation_chain.invoke(input_data).response

                print(f"Step 5: Assessing support for response {i + 1}...")
                input_data = {"response": response, "context": context}
                support = self.support_chain.invoke(input_data).response.strip().lower()
                print(f"Support assessment: {support}")

                print(f"Step 6: Evaluating utility for response {i + 1}...")
                input_data = {"query": query, "response": response}
                utility = int(self.utility_chain.invoke(input_data).response)
                print(f"Utility score: {utility}")

                responses.append((response, support, utility))

            print("Selecting the best response...")
            best_response = max(responses, key=lambda x: (x[1] == 'fully supported', x[2]))
            print(f"Best response support: {best_response[1]}, utility: {best_response[2]}")
            query_time_len = time.time() - self.start_time - memory_construction_time
            return best_response[0], relevant_contexts, memory_construction_time, query_time_len

        print("Generating without retrieval...")
        input_data = {"query": query, "context": "No retrieval necessary."}
        memory_construction_time = time.time() - self.start_time
        query_time_len = time.time() - self.start_time - memory_construction_time
        return self.generation_chain.invoke(input_data).response, "No retrieval necessary.", memory_construction_time, query_time_len
