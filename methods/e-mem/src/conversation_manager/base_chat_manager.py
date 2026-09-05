"""Base class for chat managers with shared functionality."""

import logging
from abc import abstractmethod
from typing import Any, Dict, List, Optional

import src.agent.base as base_agent_module
from src.agent.base import BaseAgent
from src.utils.prompt import AGGREGATOR_PROMPT, CHAT_SYS_PROMPT

logger = logging.getLogger(__name__)


class BaseChatManager(BaseAgent):
    """
    Abstract base class for chat managers.

    Provides shared functionality for both KV cache and text storage modes:
    - Tool definitions
    - Chat interface
    - Memory aggregation

    Subclasses must implement:
    - _create_memory_handler(): Create the appropriate memory handler
    - memory_handler property: Return the memory handler instance
    """

    # Tool definitions shared by all implementations
    ADD_MEMORY_TOOL: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "add_memory",
            "description": "Store information into memory blocks for future retrieval.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory": {
                        "type": "string",
                        "description": "The memory content to be stored.",
                    }
                },
                "required": ["memory"],
            },
        },
    }

    SEARCH_MEMORY_TOOL: Dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": "Query information from memory blocks to answer questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search memory.",
                    }
                },
                "required": ["query"],
            },
        },
    }

    def __init__(
        self,
        chat_openai_config: Dict[str, Any],
        aggregator_openai_config: Dict[str, Any],
        system_prompt: str = CHAT_SYS_PROMPT,
    ) -> None:
        """
        Initialize base chat manager.

        Args:
            chat_openai_config: OpenAI API configuration dictionary for manager/chat logic.
            aggregator_openai_config: Dedicated config for aggregation.
            system_prompt: System prompt for the chat agent.
        """
        super().__init__(chat_openai_config, system_prompt)
        self.aggregator_openai_config = aggregator_openai_config.copy()
        self.aggregator_model = self.aggregator_openai_config.pop(
            "model", "gpt-4o-mini"
        )
        self.aggregator_llm = base_agent_module.OpenAI(
            **self.aggregator_openai_config
        )
        self.last_queried_memory: Optional[str] = None
        self.auto_save: bool = False
        self.save_original_input: bool = False
        self.handle_user_input: Optional[str] = None

    @property
    @abstractmethod
    def memory_handler(self) -> Any:
        """Return the memory handler instance."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the manager name."""
        pass

    def chat(
        self,
        user_input: str,
        outer_tools: Optional[List[Dict[str, Any]]] = None,
        auto_save: bool = False,
        save_original_input: bool = False,
        max_new_tokens: int = 1024,
    ) -> str:
        """
        Chat with the user.

        Args:
            user_input: The user's input message.
            outer_tools: Additional tools to include.
            auto_save: If True, directly save input without LLM processing.
            save_original_input: If True, save original input instead of LLM-extracted content.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            The chat response.
        """
        self.handle_user_input = user_input
        self.auto_save = auto_save
        self.save_original_input = save_original_input

        # Auto-save mode: directly save without LLM processing
        if auto_save:
            logger.debug("Auto-save mode: directly saving input")
            return self.add_memory(user_input)

        # Normal mode: let LLM decide
        tools = [] if outer_tools is None else outer_tools.copy()
        tools.append(self.ADD_MEMORY_TOOL)
        tools.append(self.SEARCH_MEMORY_TOOL)

        user_prompt_formatted = f"""Please read the user input carefully and answer the question or follow the instructions to finish the tasks:
        <user_input>{user_input}</user_input>
        """

        try:
            response = self.generate_response(
                user_prompt_formatted, tools=tools, max_tokens=max_new_tokens, max_tool_rounds=1
            )
        except RuntimeError as e:
            logger.error(f"RuntimeError in generate_response: {e}", exc_info=True)
            return "Not mentioned in the conversation."

        return response

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool call.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.

        Returns:
            Tool execution result.
        """
        logger.info(f"Executing tool: {tool_name}")
        logger.debug(f"Tool arguments: {arguments}")

        if tool_name == "add_memory":
            return self.add_memory(arguments.get("memory", ""))
        elif tool_name == "query_memory":
            return self.search_memory(arguments.get("query", ""))
        else:
            logger.error(f"Unknown tool: {tool_name}")
            return f"[ERROR] Unknown tool: {tool_name}"

    def add_memory(self, memory: str) -> str:
        """
        Add memory to the memory blocks.

        Args:
            memory: The memory content to store.

        Returns:
            Success or failure message.
        """
        target_memory = self.handle_user_input if self.save_original_input else memory
        if not target_memory:
            logger.warning("No memory content provided")
            return "[ERROR] No memory content provided."

        try:
            logger.info(f"Adding memory: {target_memory[:100]}...")
            self.memory_handler.add_memory(target_memory)
            logger.info("Memory added successfully")
        except Exception as e:
            logger.error(f"Memory adding failed: {e}", exc_info=True)
            return f"[ERROR] Memory adding failed: {e}"

        return "[SUCCESS] Memory added successfully."

    def search_memory(self, query: str) -> str:
        """
        Search memory blocks with the query.

        Args:
            query: The query to search memory.

        Returns:
            The search result.
        """
        if not query:
            logger.warning("No query content provided")
            return "[ERROR] No query content provided."

        try:
            logger.info(f"Querying memory: {query}")
            raw_result = self.memory_handler.query_memory(query)
            logger.info(f"Raw memory query result length: {len(raw_result)} chars")
            logger.info(f"Memory query result: {raw_result}")

            # Aggregate results using LLM
            aggregated_result = self._aggregate_memory_results(query, raw_result)
            logger.info(f"Memory query aggregated, result length: {len(aggregated_result)} chars")
            logger.info(f"Aggregated result: {aggregated_result}")

            # Store the aggregated memory for evaluation
            self.last_queried_memory = aggregated_result

        except Exception as e:
            logger.error(f"Memory querying failed: {e}", exc_info=True)
            self.last_queried_memory = None
            return f"[ERROR] Memory querying failed: {e}"

        return aggregated_result

    def _aggregate_memory_results(self, query: str, raw_results: str) -> str:
        """
        Aggregate and summarize mixed query results.

        Args:
            query: The original query.
            raw_results: Raw results from memory blocks.

        Returns:
            Aggregated and simplified results.
        """
        prompt = AGGREGATOR_PROMPT.format(query=query, results=raw_results)

        try:
            response = self.aggregator_llm.chat.completions.create(
                model=self.aggregator_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=getattr(self, "aggregator_max_tokens", 2048),
                temperature=0,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Aggregation failed: {e}", exc_info=True)
            return raw_results  # Fallback to raw results
