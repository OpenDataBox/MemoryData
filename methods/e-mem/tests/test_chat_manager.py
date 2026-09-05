"""Tests for ChatManager."""
from unittest.mock import Mock, patch

from src.conversation_manager.chat_handler import ChatManager


class TestChatManager:
    """Test ChatManager functionality."""
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test ChatManager initialization."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        assert chat.name == "chat_manager"
        assert chat.ADD_MEMORY_TOOL["type"] == "function"
        assert chat.SEARCH_MEMORY_TOOL["type"] == "function"
        mock_memory_handler.assert_called_once()
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_chat(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test chat method."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        response = chat.chat("Hello")
        
        assert response == "Test response"
        assert chat.handle_user_input == "Hello"
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_add_memory_success(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test adding memory successfully."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        chat.handle_user_input = "Test input"
        
        result = chat.add_memory("Test memory")
        
        assert "[SUCCESS]" in result
        mock_memory_handler.add_memory.assert_called_once_with("Test memory")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_add_memory_error(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test adding memory with error."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.side_effect = Exception("Test error")
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        result = chat.add_memory("Test memory")
        
        assert "[ERROR]" in result
        assert "Test error" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_search_memory_success(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test searching memory successfully."""
        mock_memory_handler = Mock()
        mock_memory_handler.query_memory.return_value = "Found memory"
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        result = chat.search_memory("Test query")
        
        assert result == "Found memory"
        mock_memory_handler.query_memory.assert_called_once_with("Test query")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_search_memory_error(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test searching memory with error."""
        mock_memory_handler = Mock()
        mock_memory_handler.query_memory.side_effect = Exception("Query error")
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        result = chat.search_memory("Test query")
        
        assert "[ERROR]" in result
        assert "Query error" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_execute_tool_add_memory(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test executing add_memory tool."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        chat.handle_user_input = "Test input"
        
        with patch.object(chat, "add_memory", return_value="[SUCCESS]") as mock_add:
            result = chat.execute_tool("add_memory", {"memory": "Test"})
            
            assert result == "[SUCCESS]"
            mock_add.assert_called_once_with("Test")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_execute_tool_query_memory(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test executing query_memory tool."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        with patch.object(chat, "search_memory", return_value="Found") as mock_search:
            result = chat.execute_tool("query_memory", {"query": "Test"})
            
            assert result == "Found"
            mock_search.assert_called_once_with("Test")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_execute_tool_unknown(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test executing unknown tool."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        result = chat.execute_tool("unknown_tool", {})
        
        assert "[ERROR]" in result
        assert "Unknown tool" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_add_memory_empty(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test adding empty memory."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        chat.handle_user_input = ""
        
        result = chat.add_memory("")
        
        assert "[ERROR]" in result
        assert "No memory content" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_search_memory_empty_query(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test searching with empty query."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        result = chat.search_memory("")
        
        assert "[ERROR]" in result
        assert "No query content" in result
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_save_original_input(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test saving original input instead of processed."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        chat.handle_user_input = "Original input"
        chat.save_original_input = True
        
        chat.add_memory("Processed memory")
        
        mock_memory_handler.add_memory.assert_called_once_with("Original input")
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_chat_with_outer_tools(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test chat with additional outer tools."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        outer_tool = {"type": "function", "function": {"name": "custom_tool"}}
        chat.chat("Hello", outer_tools=[outer_tool])
        
        # Verify tools were passed
        call_args = mock_client.chat.completions.create.call_args
        assert len(call_args.kwargs["tools"]) == 3  # add_mem + search_mem + custom
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_enabled(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test auto_save bypasses LLM and directly saves."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        response = chat.chat("Save this information", auto_save=True)
        
        # Verify memory was saved
        mock_memory_handler.add_memory.assert_called_once_with("Save this information")
        
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()
        
        # Verify response is success message
        assert "[SUCCESS]" in response
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_with_save_original_input(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test auto_save with save_original_input flag."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.return_value = None
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        response = chat.chat(
            "Original input",
            auto_save=True,
            save_original_input=True
        )
        
        # Verify original input was saved
        mock_memory_handler.add_memory.assert_called_once_with("Original input")
        
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()
        
        assert "[SUCCESS]" in response
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_error_handling(self, mock_openai, mock_memory_handler_class, mock_openai_config):
        """Test auto_save handles errors properly."""
        mock_memory_handler = Mock()
        mock_memory_handler.add_memory.side_effect = Exception("Save failed")
        mock_memory_handler_class.return_value = mock_memory_handler
        
        mock_client = Mock()
        mock_openai.return_value = mock_client
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )
        
        response = chat.chat("Save this", auto_save=True)
        
        # Verify error message is returned
        assert "[ERROR]" in response
        assert "Save failed" in response
        
        # Verify LLM was NOT called
        mock_client.chat.completions.create.assert_not_called()
    
    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_auto_save_disabled_uses_llm(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test that auto_save=False uses normal LLM flow."""
        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "LLM response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config
        )

        response = chat.chat("Hello", auto_save=False)

        # Verify LLM was called
        mock_client.chat.completions.create.assert_called_once()

        # Verify response is from LLM
        assert response == "LLM response"

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_with_memory_segment_params(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test ChatManager initialization with max_memory_segments and max_blocks."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            max_memory_segments=10,
            max_blocks=3,
        )

        assert chat.name == "chat_manager"
        # Verify params were passed to MemoryHandler
        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["max_memory_segments"] == 10
        assert call_kwargs["max_blocks"] == 3

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_invalid_block_size_ratio(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test ChatManager raises error for invalid block_size_ratio."""
        import pytest

        with pytest.raises(ValueError, match="block_size_ratio"):
            ChatManager(
                model_id="test-model",
                chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
                block_size_ratio=0,  # Invalid: must be > 0
            )

        with pytest.raises(ValueError, match="block_size_ratio"):
            ChatManager(
                model_id="test-model",
                chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
                block_size_ratio=1.5,  # Invalid: must be <= 1
            )

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_with_all_optional_params(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test ChatManager with all optional parameters."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            clean_cache_first=False,
            model_context_window=16384,
            attn_implementation="flash_attention_2",
            device_map="cuda:0",
            router_system_prompt="Custom prompt",
            quantization_config={"load_in_4bit": True},
            max_memory={"0": "20GB"},
            offload_folder="/tmp/offload",
            overlap_mode="token",
            overlap_ratio=0.2,
            block_size_ratio=0.25,
            max_memory_segments=3,
            max_blocks=10,
        )

        assert chat.name == "chat_manager"
        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["clean_cache_first"] is False
        assert call_kwargs["model_context_window"] == 16384
        assert call_kwargs["router_system_prompt"] == "Custom prompt"
        assert call_kwargs["quantization_config"] == {"load_in_4bit": True}
        assert call_kwargs["max_memory"] == {"0": "20GB"}
        assert call_kwargs["offload_folder"] == "/tmp/offload"
        assert call_kwargs["overlap_mode"] == "token"
        assert call_kwargs["overlap_ratio"] == 0.2
        assert call_kwargs["block_size_ratio"] == 0.25

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_with_router_type_and_hybrid_config(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test ChatManager with router_type and hybrid_router_config."""
        hybrid_config = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "summary_weight": 0.4,
            "text_weight": 0.3,
            "bm25_weight": 0.3,
            "bm25_use_jieba": False,
        }
        
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            router_type="hybrid",
            hybrid_router_config=hybrid_config,
        )

        assert chat.name == "chat_manager"
        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["router_type"] == "hybrid"
        assert call_kwargs["hybrid_router_config"] == hybrid_config

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_with_llm_router_type(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test ChatManager with llm router type."""
        chat = ChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            router_type="llm",
        )

        assert chat.name == "chat_manager"
        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["router_type"] == "llm"
        assert call_kwargs["hybrid_router_config"] is None


class TestTextStorageChatManager:
    """Test TextStorageChatManager functionality."""

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test TextStorageChatManager initialization."""
        from src.conversation_manager.chat_handler import TextStorageChatManager

        chat = TextStorageChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            memory_agent_openai_config=mock_openai_config,
        )

        assert chat.name == "text_chat_manager"
        mock_memory_handler.assert_called_once()

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_with_memory_segment_params(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test TextStorageChatManager with max_memory_segments and max_blocks."""
        from src.conversation_manager.chat_handler import TextStorageChatManager

        chat = TextStorageChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            memory_agent_openai_config=mock_openai_config,
            max_memory_segments=5,
            max_blocks=8,
        )

        assert chat.name == "text_chat_manager"
        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["max_memory_segments"] == 5
        assert call_kwargs["max_blocks"] == 8

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_invalid_block_size_ratio(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test TextStorageChatManager raises error for invalid block_size_ratio."""
        import pytest

        from src.conversation_manager.chat_handler import TextStorageChatManager

        with pytest.raises(ValueError, match="block_size_ratio"):
            TextStorageChatManager(
                model_id="test-model",
                chat_openai_config=mock_openai_config,
                aggregator_openai_config=mock_openai_config,
                memory_agent_openai_config=mock_openai_config,
                block_size_ratio=0,
            )

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_chat(self, mock_openai, mock_memory_handler, mock_openai_config):
        """Test TextStorageChatManager chat method."""
        from src.conversation_manager.chat_handler import TextStorageChatManager

        mock_client = Mock()
        mock_response = Mock()
        mock_message = Mock()
        mock_message.content = "Test response"
        mock_message.tool_calls = None
        mock_response.choices = [Mock(message=mock_message)]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        chat = TextStorageChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            memory_agent_openai_config=mock_openai_config,
        )

        response = chat.chat("Hello")

        assert response == "Test response"
        assert chat.handle_user_input == "Hello"

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_init_with_router_type_and_hybrid_config(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test TextStorageChatManager with router_type and hybrid_router_config."""
        from src.conversation_manager.chat_handler import TextStorageChatManager

        hybrid_config = {
            "embedding_provider": "huggingface",
            "summary_weight": 0.3,
            "text_weight": 0.4,
            "bm25_weight": 0.3,
        }
        
        chat = TextStorageChatManager(
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            memory_agent_openai_config=mock_openai_config,
            router_type="hybrid",
            hybrid_router_config=hybrid_config,
        )

        assert chat.name == "text_chat_manager"
        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["router_type"] == "hybrid"
        assert call_kwargs["hybrid_router_config"] == hybrid_config


class TestCreateChatManagerFactory:
    """Test create_chat_manager factory function."""

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_create_kv_cache_manager(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test creating KV cache ChatManager."""
        from src.conversation_manager.factory import create_chat_manager

        manager = create_chat_manager(
            storage_mode="kv_cache",
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            router_openai_config=mock_openai_config,
        )

        assert manager.name == "chat_manager"

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_create_text_manager(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test creating text storage ChatManager."""
        from src.conversation_manager.factory import create_chat_manager

        manager = create_chat_manager(
            storage_mode="text",
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            memory_agent_openai_config=mock_openai_config,
            router_openai_config=mock_openai_config,
        )

        assert manager.name == "text_chat_manager"

    def test_create_invalid_storage_mode(self, mock_openai_config):
        """Test creating with invalid storage mode raises error."""
        import pytest

        from src.conversation_manager.factory import create_chat_manager

        with pytest.raises(ValueError, match="Invalid storage_mode"):
            create_chat_manager(
                storage_mode="invalid",
                model_id="test-model",
                chat_openai_config=mock_openai_config,
                aggregator_openai_config=mock_openai_config,
                memory_agent_openai_config=mock_openai_config,
                router_openai_config=mock_openai_config,
            )

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_text_mode_ignores_gpu_params(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test text storage mode ignores GPU-specific parameters."""
        import logging

        from src.conversation_manager.factory import create_chat_manager

        with patch.object(logging.getLogger("src.conversation_manager.factory"), "warning") as mock_warn:
            manager = create_chat_manager(
                storage_mode="text",
                model_id="test-model",
                chat_openai_config=mock_openai_config,
                aggregator_openai_config=mock_openai_config,
                memory_agent_openai_config=mock_openai_config,
                router_openai_config=mock_openai_config,
                attn_implementation="sdpa",  # GPU param, should be ignored
                device_map="auto",  # GPU param, should be ignored
            )

            assert manager.name == "text_chat_manager"
            # Verify warning was logged
            mock_warn.assert_called()

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_factory_passes_memory_segment_params(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test factory passes max_memory_segments and max_blocks."""
        from src.conversation_manager.factory import create_chat_manager

        create_chat_manager(
            storage_mode="kv_cache",
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            router_openai_config=mock_openai_config,
            max_memory_segments=7,
            max_blocks=12,
        )

        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["max_memory_segments"] == 7
        assert call_kwargs["max_blocks"] == 12

    @patch("src.conversation_manager.chat_handler.MemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_factory_passes_router_type_and_hybrid_config(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test factory passes router_type and hybrid_router_config."""
        from src.conversation_manager.factory import create_chat_manager

        hybrid_config = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "summary_weight": 0.4,
            "text_weight": 0.3,
            "bm25_weight": 0.3,
        }

        create_chat_manager(
            storage_mode="kv_cache",
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            router_openai_config=mock_openai_config,
            router_type="hybrid",
            hybrid_router_config=hybrid_config,
        )

        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["router_type"] == "hybrid"
        assert call_kwargs["hybrid_router_config"] == hybrid_config

    @patch("src.conversation_manager.chat_handler.TextMemoryHandler")
    @patch("src.agent.base.OpenAI")
    def test_factory_text_mode_passes_router_config(
        self, mock_openai, mock_memory_handler, mock_openai_config
    ):
        """Test factory passes router config for text storage mode."""
        from src.conversation_manager.factory import create_chat_manager

        hybrid_config = {
            "embedding_provider": "huggingface",
            "bm25_use_jieba": False,
        }

        create_chat_manager(
            storage_mode="text",
            model_id="test-model",
            chat_openai_config=mock_openai_config,
            aggregator_openai_config=mock_openai_config,
            memory_agent_openai_config=mock_openai_config,
            router_openai_config=mock_openai_config,
            router_type="llm",
            hybrid_router_config=hybrid_config,
        )

        call_kwargs = mock_memory_handler.call_args.kwargs
        assert call_kwargs["router_type"] == "llm"
        assert call_kwargs["hybrid_router_config"] == hybrid_config
