from _lib.core import BaseLayer, StepEvent, embody
from utils.logger import logger
from utils.stdblock import stdblock
from utils.tidgets import (
    RoleLabel,
    format_tool_args,
    loading_indicator,
    print,
)


class LoggingLayer(BaseLayer):
    "Logging adapter."

    async def handler(self, event: StepEvent) -> None:
        if self.error:
            logger.error(self.error.message)

        match event.type:
            case "agent.execution.starting":
                logger.info("Agent starting")

            case "agent.execution.started":
                logger.info("Agent started")

            case "agent.tools.executing":
                logger.info("Tool(s) executing")

            case "agent.tools.completed":
                logger.info("Tool(s) executed")

            case "agent.execution.stopped":
                logger.info("Agent stopped")

            case _ as e:
                logger.info(f"Unmatched event: '{e}'")


@embody
class VTLayer(LoggingLayer):
    """Agent with console interactivity."""

    async def percept(self):
        while True:
            user_input = stdblock(f"{RoleLabel.User} ").strip()

            if user_input.lower() == "/q":
                print(f"{RoleLabel.Agent} Goodbye!")
                break

            if user_input == "":
                user_input = "Get the current weather for New York and Los Angeles. Save the weather information to ./data/[city].txt files respectively."
                print(user_input)

            yield user_input

    async def handler(self, event: StepEvent):
        match event.type:
            case "agent.execution.starting":
                await self.suspend(fallback=loading_indicator("Ruminating..."))

            case "agent.tools.executing":
                await self.suspend(
                    fallback=loading_indicator("Calling tools..."), dwell=0.7
                )

            case "agent.tools.completed":
                for called_tool in self.called_tools:
                    tool_name = called_tool["name"]
                    tool_args = format_tool_args(called_tool["arguments"])
                    print(f"{RoleLabel.Tool} {tool_name}({tool_args})")

            case "agent.finish.stopped":
                reply = self.assistant_reply or ""
                print(
                    f"{RoleLabel.Agent} {reply}",
                    end="" if reply.endswith("\n") else "\n",
                )
