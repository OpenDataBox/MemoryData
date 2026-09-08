"""Tests for BaseAgent."""
from unittest.mock import Mock, patch

import pytest

from src.agent.base import BaseAgent


class ConcreteAgent(BaseAgent):
    """Concrete implementation for testing."""
    
    def execute_tool(self, tool_name, arguments):
        if tool_name == "test_tool":
            return f"Executed with {arguments}"
        return "Unknown tool"


class TestBaseAgent:
    """Test BaseAgent functionality."""
    
    def test_init(self, mock_openai_config):
        """Test BaseAgent initialization."""
        with patch("src.agent.base.OpenAI") as mock_openai:
            agent = ConcreteAgent(openai_config=mock_openai_config)
            
            assert agent.model == "gpt-4o-mini"
            assert agent.system_prompt == "You are a helpful assistant."
            assert len(agent.messgages) == 1
            mock_openai.assert_called_once()
    
    def test_init_no_config(self):
        """Test initialization without config raises error."""
        with pytest.raises(NotImplementedError):
            ConcreteAgent(openai_config=None)
    
    def test_generate_response_no_tools(self, mock_openai_config):
        """Test generating response without tools."""
        with patch("src.agent.base.OpenAI") as mock_openai:
            mock_client = Mock()
            mock_response = Mock()
            mock_message = Mock()
            mock_message.content = "Test response"
            mock_message.tool_calls = None
            mock_response.choices = [Mock(message=mock_message)]
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = ConcreteAgent(openai_config=mock_openai_config)
            response = agent.generate_response("Test question")
            
            assert response == "Test response"
            # After reset, should only have system message
            assert len(agent.messgages) == 1
    
    def test_generate_response_with_tool_call(self, mock_openai_config):
        """Test generating response with tool call."""
        with patch("src.agent.base.OpenAI") as mock_openai:
            mock_client = Mock()
            
            # First response with tool call
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"arg": "value"}'
            
            mock_message1 = Mock()
            mock_message1.content = None
            mock_message1.tool_calls = [mock_tool_call]
            mock_response1 = Mock()
            mock_response1.choices = [Mock(message=mock_message1)]
            
            # Second response after tool execution
            mock_message2 = Mock()
            mock_message2.content = "Final response"
            mock_message2.tool_calls = None
            mock_response2 = Mock()
            mock_response2.choices = [Mock(message=mock_message2)]
            
            mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
            mock_openai.return_value = mock_client
            
            agent = ConcreteAgent(openai_config=mock_openai_config)
            response = agent.generate_response("Test question")
            
            assert response == "Final response"
            assert mock_client.chat.completions.create.call_count == 2
    
    def test_reset(self, mock_openai_config):
        """Test reset functionality."""
        with patch("src.agent.base.OpenAI"):
            agent = ConcreteAgent(openai_config=mock_openai_config)
            agent.messgages.append({"role": "user", "content": "test"})
            
            agent.reset()
            
            assert len(agent.messgages) == 1
            assert agent.messgages[0]["role"] == "system"
    
    def test_max_tool_rounds_exceeded(self, mock_openai_config):
        """Test error when max tool rounds exceeded."""
        with patch("src.agent.base.OpenAI") as mock_openai:
            mock_client = Mock()
            
            # Always return tool calls
            mock_tool_call = Mock()
            mock_tool_call.id = "call_123"
            mock_tool_call.function.name = "test_tool"
            mock_tool_call.function.arguments = '{"arg": "value"}'
            
            mock_message = Mock()
            mock_message.content = None
            mock_message.tool_calls = [mock_tool_call]
            mock_response = Mock()
            mock_response.choices = [Mock(message=mock_message)]
            
            mock_client.chat.completions.create.return_value = mock_response
            mock_openai.return_value = mock_client
            
            agent = ConcreteAgent(openai_config=mock_openai_config)
            
            with pytest.raises(RuntimeError, match="Max tool rounds reached"):
                agent.generate_response("Test question", max_tool_rounds=2)
    
    def test_generate_response_with_multiple_tools(self, mock_openai_config):
        """Test generating response with multiple tool calls in one round."""
        with patch("src.agent.base.OpenAI") as mock_openai:
            mock_client = Mock()
            
            # First response with multiple tool calls
            mock_tool_call1 = Mock()
            mock_tool_call1.id = "call_1"
            mock_tool_call1.function.name = "test_tool"
            mock_tool_call1.function.arguments = '{"arg": "value1"}'
            
            mock_tool_call2 = Mock()
            mock_tool_call2.id = "call_2"
            mock_tool_call2.function.name = "test_tool"
            mock_tool_call2.function.arguments = '{"arg": "value2"}'
            
            mock_message1 = Mock()
            mock_message1.content = None
            mock_message1.tool_calls = [mock_tool_call1, mock_tool_call2]
            mock_response1 = Mock()
            mock_response1.choices = [Mock(message=mock_message1)]
            
            # Second response after tool execution
            mock_message2 = Mock()
            mock_message2.content = "Final response"
            mock_message2.tool_calls = None
            mock_response2 = Mock()
            mock_response2.choices = [Mock(message=mock_message2)]
            
            mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]
            mock_openai.return_value = mock_client
            
            agent = ConcreteAgent(openai_config=mock_openai_config)
            response = agent.generate_response("Test question")
            
            assert response == "Final response"
            # After reset, should only have system message
            assert len(agent.messgages) == 1
    
    def test_custom_system_prompt(self, mock_openai_config):
        """Test initialization with custom system prompt."""
        with patch("src.agent.base.OpenAI"):
            agent = ConcreteAgent(
                openai_config=mock_openai_config,
                system_prompt="Custom prompt"
            )
            
            assert agent.system_prompt == "Custom prompt"
            assert agent.messgages[0]["content"] == "Custom prompt"
