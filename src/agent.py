"""
Agent Module - Main agent loop with tool calling
"""
import re
from .llm import generate
from .tools import TOOLS, execute_tool, get_tools_description


def get_system_prompt():
    tools_desc = get_tools_description()
    return f"""You are a helpful AI assistant. You have access to the following tools:

{tools_desc}

When you need to use a tool, respond with exactly this format:
<action>tool_name</action>
<input>tool_input</input>

After receiving the tool result, provide your final response to the user.

If you don't need a tool, just respond directly to the user.
"""


def parse_tool_call(response: str):
    """Extract tool name and input from response"""
    action_match = re.search(r"<action>(.*?)</action>", response, re.DOTALL)
    input_match = re.search(r"<input>(.*?)</input>", response, re.DOTALL)
    
    if action_match and input_match:
        return action_match.group(1).strip(), input_match.group(1).strip()
    return None, None


class Agent:
    def __init__(self, max_iterations=3):
        self.max_iterations = max_iterations
        self.history = []
    
    def reset(self):
        self.history = []
    
    def run(self, user_input: str) -> str:
        """Run agent with user input"""
        messages = [
            {"role": "system", "content": get_system_prompt()}
        ]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_input})
        
        for i in range(self.max_iterations):
            response = generate(messages)
            
            # Check for tool call
            tool_name, tool_input = parse_tool_call(response)
            
            if tool_name and tool_name in TOOLS:
                # Execute tool
                tool_result = execute_tool(tool_name, tool_input)
                
                # Add tool interaction to messages
                messages.append({"role": "assistant", "content": response})
                messages.append({"role": "user", "content": f"Tool result: {tool_result}"})
                
                print(f"[🛠 Tool] {tool_name}({tool_input}) -> {tool_result}")
            else:
                # No tool call, return response
                self.history.append({"role": "user", "content": user_input})
                self.history.append({"role": "assistant", "content": response})
                return response
        
        # Max iterations reached
        return "I couldn't complete the task in the allowed steps."
