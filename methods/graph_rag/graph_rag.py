import networkx as nx
from langchain_community.vectorstores import FAISS

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

try:
    from langchain_classic.retrievers import ContextualCompressionRetriever
except ImportError:
    from langchain.retrievers import ContextualCompressionRetriever

try:
    from langchain_classic.retrievers.document_compressors import LLMChainExtractor
except ImportError:
    from langchain.retrievers.document_compressors import LLMChainExtractor

try:
    from langchain_community.callbacks.manager import get_openai_callback
except ImportError:
    from langchain.callbacks import get_openai_callback

from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import sys
from contextlib import nullcontext
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import List, Tuple, Dict
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import spacy
import heapq
import argparse
import tiktoken

from pydantic import BaseModel, Field


from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np
from spacy.cli import download
from spacy.lang.en import English

sys.path.append(os.path.abspath(
    os.path.join(os.getcwd(), '..')))  # Add the parent directory to the path sicnce we work with notebooks

# Load environment variables from a .env file
load_dotenv()

from utils.provider_utils import (
    normalize_provider as _normalize_provider,
    resolve_env_value as _resolve_env_value,
    resolve_base_url as _resolve_base_url,
    use_azure_openai as _use_azure_openai,
    create_chat_llm as _create_chat_llm,
    create_embedding_model as _create_embedding_model,
    ApproximateTokenizer,
)

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


class _NullCallback:
    total_tokens = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_cost = 0.0


def _infer_model_provider(model_name, configured_provider=None, base_url=None, base_url_env=None):
    normalized_provider = _normalize_provider(configured_provider)
    if normalized_provider:
        return normalized_provider
    if base_url or base_url_env:
        return "openai_compatible"

    normalized_model_name = (model_name or "").strip().lower()
    if "gemini" in normalized_model_name:
        return "gemini"
    if normalized_model_name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if any(hint in normalized_model_name for hint in ["qwen", "deepseek", "mistral", "moonshot", "glm", "llama", "openrouter"]):
        return "openai_compatible"
    return "openai"


def _create_tokenizer(model_name, tokenizer_model=None, tokenizer_encoding=None):
    if tokenizer_encoding:
        try:
            return tiktoken.get_encoding(tokenizer_encoding)
        except Exception:
            pass

    tokenizer_candidates = [tokenizer_model, model_name, "gpt-4o-mini"]
    seen_candidates = set()
    for candidate in tokenizer_candidates:
        if not candidate or candidate in seen_candidates:
            continue
        seen_candidates.add(candidate)
        try:
            return tiktoken.encoding_for_model(candidate)
        except Exception:
            continue

    return ApproximateTokenizer()


# Define the document processor class
# Define the DocumentProcessor class
class DocumentProcessor:
    def __init__(self, embeddings=None):
        """
        Initializes the DocumentProcessor with a text splitter and OpenAI embeddings.

        Attributes:
        - text_splitter: An instance of RecursiveCharacterTextSplitter with specified chunk size and overlap.
        - embeddings: An instance of OpenAIEmbeddings used for embedding documents.
        """
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=200)
        self.embeddings = embeddings or OpenAIEmbeddings()

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into smaller chunks and creating a vector store.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - tuple: A tuple containing:
          - splits (list of str): The list of split document chunks.
          - vector_store (FAISS): A FAISS vector store created from the split document chunks and their embeddings.
        """
        splits = self.text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, vector_store

    def create_embeddings_batch(self, texts, batch_size=32):
        """
        Creates embeddings for a list of texts in batches.

        Args:
        - texts (list of str): A list of texts to be embedded.
        - batch_size (int, optional): The number of texts to process in each batch. Default is 32.

        Returns:
        - numpy.ndarray: An array of embeddings for the input texts.
        """
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)
        return np.array(embeddings)

    def compute_similarity_matrix(self, embeddings):
        """
        Computes a cosine similarity matrix for a given set of embeddings.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the input embeddings.
        """
        return cosine_similarity(embeddings)


# Define the knowledge graph class
# Define the Concepts class
class Concepts(BaseModel):
    concepts_list: List[str] = Field(description="List of concepts")


# Define the KnowledgeGraph class
class KnowledgeGraph:
    def __init__(self):
        """
        Initializes the KnowledgeGraph with a graph, lemmatizer, and NLP model.

        Attributes:
        - graph: An instance of a networkx Graph.
        - lemmatizer: An instance of WordNetLemmatizer.
        - concept_cache: A dictionary to cache extracted concepts.
        - nlp: An instance of a spaCy NLP model.
        - edges_threshold: A float value that sets the threshold for adding edges based on similarity.
        """
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = 0.8

    def build_graph(self, splits, llm, embedding_model):
        """
        Builds the knowledge graph by adding nodes, creating embeddings, extracting concepts, and adding edges.

        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.
        - embedding_model: An instance of an embedding model.

        Returns:
        - None
        """
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits):
        """
        Adds nodes to the graph from the document splits.

        Args:
        - splits (list): A list of document splits.

        Returns:
        - None
        """
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits, embedding_model):
        """
        Creates embeddings for the document splits using the embedding model.

        Args:
        - splits (list): A list of document splits.
        - embedding_model: An instance of an embedding model.

        Returns:
        - numpy.ndarray: An array of embeddings for the document splits.
        """
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _compute_similarities(self, embeddings):
        """
        Computes the cosine similarity matrix for the embeddings.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings.

        Returns:
        - numpy.ndarray: A cosine similarity matrix for the embeddings.
        """
        return cosine_similarity(embeddings)

    def _load_spacy_model(self):
        """
        Loads the spaCy NLP model, downloading it if necessary.

        Args:
        - None

        Returns:
        - spacy.Language: An instance of a spaCy NLP model.
        """
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def _extract_concepts_and_entities(self, content, llm):
        """
        Extracts concepts and named entities from the content using spaCy and a large language model.

        Args:
        - content (str): The content from which to extract concepts and entities.
        - llm: An instance of a large language model.

        Returns:
        - list: A list of extracted concepts and entities.
        """
        if content in self.concept_cache:
            return self.concept_cache[content]

        # Extract named entities using spaCy
        doc = self.nlp(content)
        named_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]]

        # Extract general concepts using LLM
        concept_extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_extraction_prompt | llm.with_structured_output(Concepts)   
        
        ###TODO: phrase error !!!! 
        try:
            general_concepts = concept_chain.invoke({"text": content}).concepts_list
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            general_concepts = []
            
        # Combine named entities and general concepts
        all_concepts = list(set(named_entities + general_concepts))

        self.concept_cache[content] = all_concepts
        return all_concepts

    def _extract_concepts(self, splits, llm):
        """
        Extracts concepts for all document splits using multi-threading.

        Args:
        - splits (list): A list of document splits.
        - llm: An instance of a large language model.

        Returns:
        - None
        """
        #tok = tiktoken.encoding_for_model("gpt-4o-mini")
        #for i, split in enumerate(splits):
        #    print("spilit tokens", len(tok.encode(split.page_content)))
            
        with ThreadPoolExecutor() as executor:
            future_to_node = {executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                              for i, split in enumerate(splits)}

            for future in tqdm(as_completed(future_to_node), total=len(splits),
                               desc="Extracting concepts and entities"):
                node = future_to_node[future]
                concepts = future.result()
                self.graph.nodes[node]['concepts'] = concepts


    def _add_edges(self, embeddings):
        """
        Adds edges to the graph based on the similarity of embeddings and shared concepts.

        Args:
        - embeddings (numpy.ndarray): An array of embeddings for the document splits.

        Returns:
        - None
        """
        similarity_matrix = self._compute_similarities(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = set(self.graph.nodes[node1]['concepts']) & set(
                        self.graph.nodes[node2]['concepts'])
                    edge_weight = self._calculate_edge_weight(node1, node2, similarity_score, shared_concepts)
                    self.graph.add_edge(node1, node2, weight=edge_weight,
                                        similarity=similarity_score,
                                        shared_concepts=list(shared_concepts))

    def _calculate_edge_weight(self, node1, node2, similarity_score, shared_concepts, alpha=0.7, beta=0.3):
        """
        Calculates the weight of an edge based on similarity score and shared concepts.

        Args:
        - node1 (int): The first node.
        - node2 (int): The second node.
        - similarity_score (float): The similarity score between the nodes.
        - shared_concepts (set): The set of shared concepts between the nodes.
        - alpha (float, optional): The weight of the similarity score. Default is 0.7.
        - beta (float, optional): The weight of the shared concepts. Default is 0.3.

        Returns:
        - float: The calculated weight of the edge.
        """
        max_possible_shared = min(len(self.graph.nodes[node1]['concepts']), len(self.graph.nodes[node2]['concepts']))
        normalized_shared_concepts = len(shared_concepts) / max_possible_shared if max_possible_shared > 0 else 0
        return alpha * similarity_score + beta * normalized_shared_concepts

    def _lemmatize_concept(self, concept):
        """
        Lemmatizes a given concept.

        Args:
        - concept (str): The concept to be lemmatized.

        Returns:
        - str: The lemmatized concept.
        """
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])


# Define the Query Engine class
# Define the AnswerCheck class
class AnswerCheck(BaseModel):
    is_complete: bool = Field(description="Whether the current context provides a complete answer to the query")
    answer: str = Field(description="The current answer based on the context, if any")


# Define the QueryEngine class
class QueryEngine:
    def __init__(self, vector_store, knowledge_graph, llm, control_llm=None, retrieve_num=5,
                 tokenizer=None, enable_openai_callback=True):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        # Use a separate shorter-budget LLM for internal structured/control steps
        # so benchmark answer limits do not truncate concept extraction or checks.
        self.control_llm = control_llm or llm
        #self.max_context_length = 4000
        self.retrieve_num = retrieve_num
        self.tokenizer = tokenizer or ApproximateTokenizer()
        self.enable_openai_callback = enable_openai_callback
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """
        Creates a chain to check if the context provides a complete answer to the query.

        Args:
        - None

        Returns:
        - Chain: A chain to check if the context provides a complete answer.
        """
        answer_check_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template="Given the query: '{query}'\n\nAnd the current context:\n{context}\n\nDoes this context provide a complete answer to the query? If yes, provide the answer. If no, state that the answer is incomplete.\n\nIs complete answer (Yes/No):\nAnswer (if complete):"
        )
        return answer_check_prompt | self.control_llm.with_structured_output(AnswerCheck)

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """
        Checks if the current context provides a complete answer to the query.

        Args:
        - query (str): The query to be answered.
        - context (str): The current context.

        Returns:
        - tuple: A tuple containing:
          - is_complete (bool): Whether the context provides a complete answer.
          - answer (str): The answer based on the context, if complete.
        """
        try:
            response = self.answer_check_chain.invoke({"query": query, "context": context})
        except Exception as e:
            print(f"Error extracting concepts: {e}")
            response = AnswerCheck(is_complete=False, answer="")
            
        return response.is_complete, response.answer

    def _expand_context(self, query: str, relevant_docs) -> Tuple[str, List[int], Dict[int, str], str]:
        """
        Expands the context by traversing the knowledge graph using a Dijkstra-like approach.

        This method implements a modified version of Dijkstra's algorithm to explore the knowledge graph,
        prioritizing the most relevant and strongly connected information. The algorithm works as follows:

        1. Initialize:
           - Start with nodes corresponding to the most relevant documents.
           - Use a priority queue to manage the traversal order, where priority is based on connection strength.
           - Maintain a dictionary of best known "distances" (inverse of connection strengths) to each node.

        2. Traverse:
           - Always explore the node with the highest priority (strongest connection) next.
           - For each node, check if we've found a complete answer.
           - Explore the node's neighbors, updating their priorities if a stronger connection is found.

        3. Concept Handling:
           - Track visited concepts to guide the exploration towards new, relevant information.
           - Expand to neighbors only if they introduce new concepts.

        4. Termination:
           - Stop if a complete answer is found.
           - Continue until the priority queue is empty (all reachable nodes explored).

        This approach ensures that:
        - We prioritize the most relevant and strongly connected information.
        - We explore new concepts systematically.
        - We find the most relevant answer by following the strongest connections in the knowledge graph.

        Args:
        - query (str): The query to be answered.
        - relevant_docs (List[Document]): A list of relevant documents to start the traversal.

        Returns:
        - tuple: A tuple containing:
          - expanded_context (str): The accumulated context from traversed nodes.
          - traversal_path (List[int]): The sequence of node indices visited.
          - filtered_content (Dict[int, str]): A mapping of node indices to their content.
          - final_answer (str): The final answer found, if any.
        """
        # Initialize variables
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}  # Stores the best known "distance" (inverse of connection strength) to each node

        print("\nTraversing the knowledge graph:")

        # Initialize priority queue with closest nodes from relevant docs
        for doc in relevant_docs:
            # Find the most similar node in the knowledge graph for each relevant document
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            closest_node_content, similarity_score = closest_nodes[0]

            # Get the corresponding node in our knowledge graph
            closest_node = next(n for n in self.knowledge_graph.graph.nodes if
                                self.knowledge_graph.graph.nodes[n]['content'] == closest_node_content.page_content)

            # Initialize priority (inverse of similarity score for min-heap behavior)
            priority = 1 / similarity_score
            heapq.heappush(priority_queue, (priority, closest_node))
            distances[closest_node] = priority

        step = 0
        while priority_queue:
            # Get the node with the highest priority (lowest distance value)
            current_priority, current_node = heapq.heappop(priority_queue)

            # Skip if we've already found a better path to this node
            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                step += 1
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node]['content']
                node_concepts = self.knowledge_graph.graph.nodes[current_node]['concepts']

                # Add node content to our accumulated context
                filtered_content[current_node] = node_content
                expanded_context += "\n" + node_content if expanded_context else node_content

                # Log the current step for debugging and visualization
                print(f"\nStep {step} - Node {current_node}:")
                print(f"Content: {node_content[:100]}...")
                print(f"Concepts: {', '.join(node_concepts)}")
                print("-" * 50)

                # Check if we have a complete answer with the current context
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                # Process the concepts of the current node
                node_concepts_set = set(self.knowledge_graph._lemmatize_concept(c) for c in node_concepts)
                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    # Explore neighbors
                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data['weight']

                        # Calculate new distance (priority) to the neighbor
                        # Note: We use 1 / edge_weight because higher weights mean stronger connections
                        distance = current_priority + (1 / edge_weight)

                        # If we've found a stronger connection to the neighbor, update its distance
                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

                            # Process the neighbor node if it's not already in our traversal path
                            if neighbor not in traversal_path:
                                step += 1
                                traversal_path.append(neighbor)
                                neighbor_content = self.knowledge_graph.graph.nodes[neighbor]['content']
                                neighbor_concepts = self.knowledge_graph.graph.nodes[neighbor]['concepts']

                                filtered_content[neighbor] = neighbor_content
                                expanded_context += "\n" + neighbor_content if expanded_context else neighbor_content

                                # Log the neighbor node information
                                print(f"\nStep {step} - Node {neighbor} (neighbor of {current_node}):")
                                print(f"Content: {neighbor_content[:100]}...")
                                print(f"Concepts: {', '.join(neighbor_concepts)}")
                                print("-" * 50)

                                # Check if we have a complete answer after adding the neighbor's content
                                is_complete, answer = self._check_answer(query, expanded_context)
                                if is_complete:
                                    final_answer = answer
                                    break

                                # Process the neighbor's concepts
                                neighbor_concepts_set = set(
                                    self.knowledge_graph._lemmatize_concept(c) for c in neighbor_concepts)
                                if not neighbor_concepts_set.issubset(visited_concepts):
                                    visited_concepts.update(neighbor_concepts_set)

                # If we found a final answer, break out of the main loop
                if final_answer:
                    break

        # If we haven't found a complete answer, generate one using the LLM
        if not final_answer:
            print("\nGenerating final answer...")
            response_prompt = PromptTemplate(
                input_variables=["query", "context"],
                template="Based on the following context, please answer the query very concisely.\n\nContext: {context}\n\nQuery: {query}"
            )
            response_chain = response_prompt | self.llm
            
            ### Not exceed 128K
            tokenzier = self.tokenizer
            llm_length_limit = 127000
            if len(tokenzier.encode(expanded_context)) > llm_length_limit:
                print("\n\n\n(545) The context is too long, we need to truncate it.\n\n\n")
                encoded = tokenzier.encode(expanded_context)
                expanded_context = tokenzier.decode(encoded[llm_length_limit:])
            ###
            input_data = {"query": query, "context": expanded_context}   
            final_answer = response_chain.invoke(input_data)

        return expanded_context, traversal_path, filtered_content, final_answer

    def query(self, query: str) -> Tuple[str, List[int], Dict[int, str]]:
        """
        Processes a query by retrieving relevant documents, expanding the context, and generating the final answer.

        Args:
        - query (str): The query to be answered.

        Returns:
        - tuple: A tuple containing:
          - final_answer (str): The final answer to the query.
          - traversal_path (list): The traversal path of nodes in the knowledge graph.
          - filtered_content (dict): The filtered content of nodes.
        """
        ## find the "Question" in the message
        import re
        match = re.search(r"Now Answer the Question:\s*(.*)", query, re.DOTALL)
        if match:
            retrieval_query =  ''.join(match.groups())
        else:
            match = re.search(r"Here is the conversation:\s*(.*)", query, re.DOTALL)
            if match:
                retrieval_query =  ''.join(match.groups())
            else:
                retrieval_query = query
        ## End of newly added code
        callback_context = get_openai_callback() if self.enable_openai_callback else nullcontext(_NullCallback())
        with callback_context as cb:
            print(f"\nRetrieval query: {retrieval_query}")
            relevant_docs = self._retrieve_relevant_documents(retrieval_query)
            expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(query,
                                                                                                    relevant_docs)

            if not final_answer:
                print("\nGenerating final answer...")
                response_prompt = PromptTemplate(
                    input_variables=["query", "context"],
                    template="Based on the following context, please answer the query very concisely.\n\nContext: {context}\n\nQuery: {query}"
                )

                response_chain = response_prompt | self.llm
                ### Not exceed 128K
                tokenzier = self.tokenizer
                llm_length_limit = 127000
                if len(tokenzier.encode(expanded_context)) > llm_length_limit:
                    print("\n\n\n(592) The context is too long, we need to truncate it.\n\n\n")
                    encoded = tokenzier.encode(expanded_context)
                    expanded_context = tokenzier.decode(encoded[llm_length_limit:])
                ###
                input_data = {"query": query, "context": expanded_context}
                response = response_chain.invoke(input_data)
                final_answer = response
            else:
                print("\nComplete answer found during traversal.")

            print(f"\nFinal Answer: {final_answer}")
            print(f"\nTotal Tokens: {cb.total_tokens}")
            print(f"Prompt Tokens: {cb.prompt_tokens}")
            print(f"Completion Tokens: {cb.completion_tokens}")
            print(f"Total Cost (USD): ${cb.total_cost}")
            
        # add expanded_context here
        return final_answer, expanded_context, traversal_path, filtered_content

    def _retrieve_relevant_documents(self, query: str):
        """
        Retrieves relevant documents based on the query using the vector store.

        Args:
        - query (str): The query to be answered.

        Returns:
        - list: A list of relevant documents.
        """
        print("\nRetrieving relevant documents...")
        retriever = self.vector_store.as_retriever(search_type="similarity", search_kwargs={"k": self.retrieve_num})
        compressor = LLMChainExtractor.from_llm(self.control_llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever)
        
        return compression_retriever.invoke(query)


# Import necessary libraries
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# Define the Visualizer class
class Visualizer:
    @staticmethod
    def visualize_traversal(graph, traversal_path):
        """
        Visualizes the traversal path on the knowledge graph with nodes, edges, and traversal path highlighted.

        Args:
        - graph (networkx.Graph): The knowledge graph containing nodes and edges.
        - traversal_path (list of int): The list of node indices representing the traversal path.

        Returns:
        - None
        """
        traversal_graph = nx.DiGraph()

        # Add nodes and edges from the original graph
        for node in graph.nodes():
            traversal_graph.add_node(node)
        for u, v, data in graph.edges(data=True):
            traversal_graph.add_edge(u, v, **data)

        fig, ax = plt.subplots(figsize=(16, 12))

        # Generate positions for all nodes
        pos = nx.spring_layout(traversal_graph, k=1, iterations=50)

        # Draw regular edges with color based on weight
        edges = traversal_graph.edges()
        edge_weights = [traversal_graph[u][v].get('weight', 0.5) for u, v in edges]
        nx.draw_networkx_edges(traversal_graph, pos,
                               edgelist=edges,
                               edge_color=edge_weights,
                               edge_cmap=plt.cm.Blues,
                               width=2,
                               ax=ax)

        # Draw nodes
        nx.draw_networkx_nodes(traversal_graph, pos,
                               node_color='lightblue',
                               node_size=3000,
                               ax=ax)

        # Draw traversal path with curved arrows
        edge_offset = 0.1
        for i in range(len(traversal_path) - 1):
            start = traversal_path[i]
            end = traversal_path[i + 1]
            start_pos = pos[start]
            end_pos = pos[end]

            # Calculate control point for curve
            mid_point = ((start_pos[0] + end_pos[0]) / 2, (start_pos[1] + end_pos[1]) / 2)
            control_point = (mid_point[0] + edge_offset, mid_point[1] + edge_offset)

            # Draw curved arrow
            arrow = patches.FancyArrowPatch(start_pos, end_pos,
                                            connectionstyle=f"arc3,rad={0.3}",
                                            color='red',
                                            arrowstyle="->",
                                            mutation_scale=20,
                                            linestyle='--',
                                            linewidth=2,
                                            zorder=4)
            ax.add_patch(arrow)

        # Prepare labels for the nodes
        labels = {}
        for i, node in enumerate(traversal_path):
            concepts = graph.nodes[node].get('concepts', [])
            label = f"{i + 1}. {concepts[0] if concepts else ''}"
            labels[node] = label

        for node in traversal_graph.nodes():
            if node not in labels:
                concepts = graph.nodes[node].get('concepts', [])
                labels[node] = concepts[0] if concepts else ''

        # Draw labels
        nx.draw_networkx_labels(traversal_graph, pos, labels, font_size=8, font_weight="bold", ax=ax)

        # Highlight start and end nodes
        start_node = traversal_path[0]
        end_node = traversal_path[-1]

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[start_node],
                               node_color='lightgreen',
                               node_size=3000,
                               ax=ax)

        nx.draw_networkx_nodes(traversal_graph, pos,
                               nodelist=[end_node],
                               node_color='lightcoral',
                               node_size=3000,
                               ax=ax)

        ax.set_title("Graph Traversal Flow")
        ax.axis('off')

        # Add colorbar for edge weights
        sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues,
                                   norm=plt.Normalize(vmin=min(edge_weights), vmax=max(edge_weights)))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Edge Weight', rotation=270, labelpad=15)

        # Add legend
        regular_line = plt.Line2D([0], [0], color='blue', linewidth=2, label='Regular Edge')
        traversal_line = plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Traversal Path')
        start_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15,
                                 label='Start Node')
        end_point = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', markersize=15,
                               label='End Node')
        legend = plt.legend(handles=[regular_line, traversal_line, start_point, end_point], loc='upper left',
                            bbox_to_anchor=(0, 1), ncol=2)
        legend.get_frame().set_alpha(0.8)

        plt.tight_layout()
        plt.show()

    @staticmethod
    def print_filtered_content(traversal_path, filtered_content):
        """
        Prints the filtered content of visited nodes in the order of traversal.

        Args:
        - traversal_path (list of int): The list of node indices representing the traversal path.
        - filtered_content (dict of int: str): A dictionary mapping node indices to their filtered content.

        Returns:
        - None
        """
        print("\nFiltered content of visited nodes in order of traversal:")
        for i, node in enumerate(traversal_path):
            print(f"\nStep {i + 1} - Node {node}:")
            print(
                f"Filtered Content: {filtered_content.get(node, 'No filtered content available')[:200]}...")  # Print first 200 characters
            print("-" * 50)


# Define the graph RAG class
class GraphRAG:
    def __init__(self, temperature=0.7, model_name="gpt-4o-mini", retrieve_num=5, max_tokens=100,
                 internal_max_tokens=None,
                 provider=None, base_url=None, base_url_env=None, api_key_env=None,
                 azure_endpoint=None, azure_api_version=None, embedding_model=None,
                 embedding_provider=None, embedding_base_url=None, embedding_base_url_env=None,
                 embedding_api_key_env=None, embedding_azure_endpoint=None,
                 embedding_azure_api_version=None, tokenizer_model=None, tokenizer_encoding=None):
        """
        Initializes the GraphRAG system with components for document processing, knowledge graph construction,
        querying, and visualization.

        Args:
        - temperature (float): The temperature value for the large language model (default is 0.7).
        - model_name (str): The name of the large language model to use. 
                           OpenAI models: "gpt-4o-mini", "gpt-4", "gpt-3.5-turbo", etc.
                           Gemini models: "gemini-pro", "gemini-2.0-flash", "gemini-1.5-pro", etc.
        - retrieve_num (int): Number of documents to retrieve (default is 5).
        - max_tokens (int): The maximum number of tokens for the large language model (default is 100).

        Attributes:
        - llm: An instance of a large language model (LLM) for generating responses.
        - embedding_model: An instance of an embedding model for document embeddings.
        - document_processor: An instance of the DocumentProcessor class for processing documents.
        - knowledge_graph: An instance of the KnowledgeGraph class for building and managing the knowledge graph.
        - query_engine: An instance of the QueryEngine class for handling queries (initialized as None).
        - visualizer: An instance of the Visualizer class for visualizing the knowledge graph traversal.
        """
        self.retrieve_num = retrieve_num
        self.model_name = model_name
        self.provider = _infer_model_provider(
            model_name,
            configured_provider=provider,
            base_url=base_url,
            base_url_env=base_url_env,
        )
        self.tokenizer = _create_tokenizer(
            model_name=model_name,
            tokenizer_model=tokenizer_model,
            tokenizer_encoding=tokenizer_encoding,
        )
        self.internal_max_tokens = internal_max_tokens or max_tokens
        self.llm = _create_chat_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            provider=self.provider,
            base_url=base_url,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )
        self.control_llm = _create_chat_llm(
            model=model_name,
            temperature=temperature,
            max_tokens=self.internal_max_tokens,
            provider=self.provider,
            base_url=base_url,
            base_url_env=base_url_env,
            api_key_env=api_key_env,
            azure_endpoint=azure_endpoint,
            azure_api_version=azure_api_version,
        )
        self.embedding_model = _create_embedding_model(
            model=embedding_model,
            provider=embedding_provider,
            base_url=embedding_base_url,
            base_url_env=embedding_base_url_env,
            api_key_env=embedding_api_key_env,
            azure_endpoint=embedding_azure_endpoint,
            azure_api_version=embedding_azure_api_version,
        )
        self.document_processor = DocumentProcessor(embeddings=self.embedding_model)
        self.knowledge_graph = KnowledgeGraph()
        self.query_engine = None
        self.visualizer = Visualizer()
        #self.process_documents(documents)

    def _is_gemini_model(self, model_name: str) -> bool:
        """
        Check if the model name corresponds to a Gemini model.
        
        Args:
        - model_name (str): The model name to check.
        
        Returns:
        - bool: True if it's a Gemini model, False otherwise.
        """
        gemini_patterns = [
            'gemini',
            'models/gemini'  # Full model path format
        ]
        model_name_lower = model_name.lower()
        return any(pattern in model_name_lower for pattern in gemini_patterns)
    
    def _is_openai_model(self, model_name: str) -> bool:
        """
        Check if the model name corresponds to an OpenAI model.
        
        Args:
        - model_name (str): The model name to check.
        
        Returns:
        - bool: True if it's an OpenAI model, False otherwise.
        """
        openai_patterns = [
            'gpt',
            'text-davinci',
            'text-curie',
            'text-babbage',
            'text-ada',
            'o1',  # OpenAI o1 models
            'claude'  # In case someone uses this with Claude through OpenAI-compatible APIs
        ]
        model_name_lower = model_name.lower()
        return any(pattern in model_name_lower for pattern in openai_patterns)

    def process_documents(self, documents):
        """
        Processes a list of documents by splitting them into chunks, embedding them, and building a knowledge graph.

        Args:
        - documents (list of str): A list of documents to be processed.

        Returns:
        - None
        """
        splits, vector_store = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.control_llm, self.embedding_model)
        self.query_engine = QueryEngine(
            vector_store,
            self.knowledge_graph,
            self.llm,
            self.control_llm,
            retrieve_num=self.retrieve_num,
            tokenizer=self.tokenizer,
            enable_openai_callback=self.provider in {"openai", "azure_openai"},
        )

    def query(self, query: str):
        """
        Handles a query by retrieving relevant information from the knowledge graph and visualizing the traversal path.

        Args:
        - query (str): The query to be answered.

        Returns:
        - str: The response to the query.
        """
        final_answer, expanded_context, traversal_path, filtered_content = self.query_engine.query(query)

        if traversal_path and False:
            self.visualizer.visualize_traversal(self.knowledge_graph.graph, traversal_path)
        else:
            print("No traversal path to visualize.")

        return final_answer, expanded_context


if __name__ == '__main__':
    pass 
    # Load the documents
    #documents = " "

    # Create a graph RAG instance
    #graph_rag = GraphRAG(documents)

    # Process the documents and create the graph
    #graph_rag.process_documents(documents)

    # Input a query and get the retrieved information from the graph RAG
    #response = graph_rag.query(args.query)
