import json
from abc import ABC, abstractmethod

from openai import OpenAI


class BaseAgent(ABC):
    def __init__(self,openai_config:dict=None,system_prompt:str="You are a helpful assistant.")->None:
        if openai_config is None:
            raise NotImplementedError("Other types of LLM provider is not supported yet.")
        
        self.openai_config=openai_config.copy()
        self.model = self.openai_config.pop("model", "gpt-4o-mini")  # Extract model, default to gpt-4o-mini
        self.system_prompt=system_prompt
        self.llm = OpenAI(**self.openai_config)
        self.messgages=[
                {"role": "system", "content": self.system_prompt}
            ]

    @abstractmethod
    def execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call. Must be implemented by subclasses."""
        pass

    def generate_response(self, question: str, max_tokens: int = 1024, tools: list = None, max_tool_rounds: int = 5,
                          temperature: float = 0, repetition_penalty: float = None) -> str:
        if tools is None:
            tools = []

        self.messgages.append({"role": "user", "content": question})

        # Build extra_body only if repetition_penalty is set
        extra_body = {}
        if repetition_penalty is not None:
            extra_body["repetition_penalty"] = repetition_penalty

        tool_round_count = 0

        while tool_round_count < max_tool_rounds:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=self.messgages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_body=extra_body if extra_body else None
            )
            
            response_message = response.choices[0].message
            self.messgages.append(response_message)
            
            # If no tool calls, return final response
            if not response_message.tool_calls:
                response_content = response_message.content
                self.reset()
                return response_content
            
            # Execute all tool calls in this round
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                function_response = self.execute_tool(function_name, function_args)
                
                self.messgages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response
                })
            
            tool_round_count += 1
        
        # Max tool rounds reached, make final call for response
        final_response = self.llm.chat.completions.create(
            model=self.model,
            messages=self.messgages,
            max_tokens=max_tokens,
            temperature=temperature,
            extra_body=extra_body if extra_body else None
        )
        
        # Check if still has tool calls after max rounds
        if final_response.choices[0].message.tool_calls:
            self.reset()
            raise RuntimeError("Max tool rounds reached but model still requesting tool calls. Increase max_tool_rounds.")
        
        response_content = final_response.choices[0].message.content
        self.reset()
        return response_content
    
    def reset(self):
        """Reset message history to initial state."""
        self.messgages = [{"role": "system", "content": self.system_prompt}]