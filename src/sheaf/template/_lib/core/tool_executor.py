import json
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Callable, TypedDict, cast

from openai.types.chat import (
    ChatCompletionMessageFunctionToolCall,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolMessageParam,
)


class CalledTool(TypedDict):
    """Metadata and results from a tool execution."""

    name: str
    arguments: str
    results: ChatCompletionToolMessageParam


# NOTE: Deprecated ToolExecutor (replaced by ToolExecution).
# if False:
#     class ToolExecutor:
#         """Handles tool execution with parallel processing."""
#
#         def __init__(self, tool_registry: dict[str, Callable[..., Any]]):
#             self._tool_registry = tool_registry
#
#         @classmethod
#         def from_callables(cls, callables: list[Callable[..., Any]]) -> "ToolExecutor":
#             """Convenience constructor from a list of functions."""
#             registry = {fn.__name__: fn for fn in callables}
#             return cls(registry)
#
#         def call_tools(
#             self, tool_calls: list[ChatCompletionMessageToolCallUnion]
#         ) -> list[CalledTool]:
#             """Execute multiple tool calls in parallel and maintain original order.
#
#             Args:
#                 tool_calls: The list of tool calls from the assistant message
#
#             Returns:
#                 List of CalledTool dicts with name, arguments, and results
#             """
#             with ThreadPoolExecutor() as pool:
#                 # Submit all tool calls for parallel execution
#                 tool_call_futures_dict: dict[
#                     Future[ChatCompletionToolMessageParam],
#                     tuple[int, str, str],
#                 ] = {
#                     pool.submit(self._execute_tool, tool): (
#                         tool_index,
#                         tool_name := getattr(
#                             getattr(tool, "function", None), "name", "unknown"
#                         ),
#                         tool_arguments := getattr(
#                             getattr(tool, "function", None), "arguments", "{}"
#                         )
#                         or "{}",
#                     )
#                     for tool_index, tool in enumerate(tool_calls)
#                 }
#
#                 # Prepare results list to maintain original order
#                 original_tool_call_order_results = cast(
#                     list[CalledTool], [None] * len(tool_calls)
#                 )
#
#                 # Collect results as they complete
#                 for completed_tool_call_future in as_completed(tool_call_futures_dict):
#                     original_tool_order_index, tool_name, tool_arguments = (
#                         tool_call_futures_dict[completed_tool_call_future]
#                     )
#                     tool_message_response = completed_tool_call_future.result()
#                     original_tool_call_order_results[original_tool_order_index] = {
#                         "name": tool_name,
#                         "arguments": tool_arguments,
#                         "results": tool_message_response,
#                     }
#
#                 return original_tool_call_order_results
#
#         def _execute_tool(
#             self,
#             tool: ChatCompletionMessageToolCallUnion,
#         ) -> ChatCompletionToolMessageParam:
#             """Execute a single tool call and return a tool message.
#
#             Args:
#                 tool: The tool call from the assistant
#
#             Returns:
#                 ChatCompletionToolMessageParam with the tool result
#             """
#             tool_id = tool.id
#             # Validate tool type
#             if getattr(tool, "type", None) != "function":
#                 print(f"Unsupported tool call: {tool_id}")
#                 return {
#                     "role": "tool",
#                     "tool_call_id": tool_id,
#                     "content": f"Error: {tool_id} is an unsupported tool call type, expected `function`.",
#                 }
#
#             tool = cast(ChatCompletionMessageFunctionToolCall, tool)
#             tool_name = tool.function.name
#             tool_arguments = tool.function.arguments or "{}"
#
#             # Parse arguments
#             try:
#                 tool_arguments_dict = json.loads(tool_arguments) if tool_arguments else {}
#             except Exception as e:
#                 print(f"Failed to parse tool arguments for {tool_name}: {e}")
#                 return {
#                     "role": "tool",
#                     "tool_call_id": tool_id,
#                     "content": "Error: Failed to parse tool arguments.",
#                 }
#
#             # Look up tool function
#             tool_function = self._tool_registry.get(tool_name)
#             if tool_function is None:
#                 print(f"Error: tool {tool_name} not available.")
#                 return {
#                     "role": "tool",
#                     "tool_call_id": tool_id,
#                     "content": f"Error: tool `{tool_name}` not available.",
#                 }
#
#             # Execute tool
#             try:
#                 tool_result = tool_function(**tool_arguments_dict)
#                 tool_result_str = (
#                     tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
#                 )
#             except Exception as e:
#                 print(f"Tool {tool_name} raised: {e}")
#                 return {
#                     "role": "tool",
#                     "tool_call_id": tool_id,
#                     "content": str(e),
#                 }
#
#             return {
#                 "role": "tool",
#                 "tool_call_id": tool_id,
#                 "content": tool_result_str,
#             }


class ToolExecution:
    """Tool execution support for BaseLayer."""

    def _init_tools(self, callables: list[Callable[..., Any]]) -> None:
        self._tool_registry = {fn.__name__: fn for fn in callables}

    def call_tools(
        self, tool_calls: list[ChatCompletionMessageToolCallUnion]
    ) -> list[CalledTool]:
        """Execute multiple tool calls in parallel and maintain original order."""
        with ThreadPoolExecutor() as pool:
            tool_call_futures_dict: dict[
                Future[ChatCompletionToolMessageParam],
                tuple[int, str, str],
            ] = {
                pool.submit(self._execute_tool, tool): (
                    tool_index,
                    tool_name := getattr(
                        getattr(tool, "function", None), "name", "unknown"
                    ),
                    tool_arguments := getattr(
                        getattr(tool, "function", None), "arguments", "{}"
                    )
                    or "{}",
                )
                for tool_index, tool in enumerate(tool_calls)
            }

            original_tool_call_order_results = cast(
                list[CalledTool], [None] * len(tool_calls)
            )

            for completed_tool_call_future in as_completed(tool_call_futures_dict):
                original_tool_order_index, tool_name, tool_arguments = (
                    tool_call_futures_dict[completed_tool_call_future]
                )
                tool_message_response = completed_tool_call_future.result()
                original_tool_call_order_results[original_tool_order_index] = {
                    "name": tool_name,
                    "arguments": tool_arguments,
                    "results": tool_message_response,
                }

            return original_tool_call_order_results

    def _execute_tool(
        self,
        tool: ChatCompletionMessageToolCallUnion,
    ) -> ChatCompletionToolMessageParam:
        tool_id = tool.id
        if getattr(tool, "type", None) != "function":
            print(f"Unsupported tool call: {tool_id}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Error: {tool_id} is an unsupported tool call type, expected `function`.",
            }

        tool = cast(ChatCompletionMessageFunctionToolCall, tool)
        tool_name = tool.function.name
        tool_arguments = tool.function.arguments or "{}"

        try:
            tool_arguments_dict = json.loads(tool_arguments) if tool_arguments else {}
        except Exception as e:
            print(f"Failed to parse tool arguments for {tool_name}: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": "Error: Failed to parse tool arguments.",
            }

        tool_function = self._tool_registry.get(tool_name)
        if tool_function is None:
            print(f"Error: tool {tool_name} not available.")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": f"Error: tool `{tool_name}` not available.",
            }

        try:
            tool_result = tool_function(**tool_arguments_dict)
            tool_result_str = (
                tool_result if isinstance(tool_result, str) else json.dumps(tool_result)
            )
        except Exception as e:
            print(f"Tool {tool_name} raised: {e}")
            return {
                "role": "tool",
                "tool_call_id": tool_id,
                "content": str(e),
            }

        return {
            "role": "tool",
            "tool_call_id": tool_id,
            "content": tool_result_str,
        }
