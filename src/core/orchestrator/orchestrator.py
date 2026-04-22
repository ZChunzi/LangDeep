"""Flow orchestrator with Supervisor pattern."""
from typing import TypedDict, Annotated, Sequence, Dict, Any, List, Optional
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
import operator
import json

# Import registries
from ..registry.model_registry import model_registry
from ..registry.tool_registry import tool_registry
from ..registry.agent_registry import agent_registry
from ..prompt.prompt_loader import MarkdownPromptLoader  # 修改：导入类，而非全局单例


class OrchestratorState(TypedDict):
    """Flow orchestrator state definition"""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    next: str
    current_task: str
    task_context: Dict[str, Any]
    agent_results: Annotated[Dict[str, Any], lambda x, y: {**x, **y}]
    workflow_plan: Optional[List[Dict[str, Any]]]
    error_count: int
    max_retries: int


class FlowOrchestrator:
    def __init__(
        self,
        supervisor_model: str = "gpt4o",
        max_retries: int = 3,
        enable_checkpoint: bool = True,
        prompt_dir: Optional[str] = None  # 新增：支持自定义 Prompt 目录
    ):
        self.supervisor_model = supervisor_model
        self.max_retries = max_retries
        self.checkpointer = MemorySaver() if enable_checkpoint else None
        self.prompt_loader = MarkdownPromptLoader(prompt_dir)  # 新增：实例化 Prompt 加载器
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(OrchestratorState)

        workflow.add_node("supervisor", self._supervisor_node)
        workflow.add_node("planner", self._planner_node)
        workflow.add_node("executor", self._executor_node)
        workflow.add_node("aggregator", self._aggregator_node)

        for agent_name in agent_registry.list_agents():
            workflow.add_node(agent_name, self._create_agent_node(agent_name))

        workflow.add_edge(START, "supervisor")
        workflow.add_conditional_edges(
            "supervisor",
            self._route_from_supervisor,
            {
                "planner": "planner",
                "executor": "executor",
                "aggregator": "aggregator",
                "end": END,
                **{name: name for name in agent_registry.list_agents()}
            }
        )

        workflow.add_edge("planner", "supervisor")
        workflow.add_edge("executor", "supervisor")
        workflow.add_conditional_edges("aggregator", self._should_end, {"supervisor": "supervisor", "end": END})

        for agent_name in agent_registry.list_agents():
            workflow.add_edge(agent_name, "supervisor")

        if self.checkpointer:
            return workflow.compile(checkpointer=self.checkpointer)
        return workflow.compile()

    def _supervisor_node(self, state: OrchestratorState) -> Dict[str, Any]:
        llm = model_registry.get_model(self.supervisor_model)
        supervisor_prompt = self.prompt_loader.load_prompt("supervisor")  # 修改：使用实例加载器

        messages = state["messages"]
        task_context = state.get("task_context", {})
        agent_results = state.get("agent_results", {})
        workflow_plan = state.get("workflow_plan")

        available_agents = []
        for name in agent_registry.list_agents():
            meta = agent_registry.get_metadata(name)
            available_agents.append({
                "name": name,
                "description": meta.description if meta else "",
                "capabilities": meta.capabilities if meta else []
            })

        valid_targets = ["planner", "executor", "aggregator", "end"] + [a["name"] for a in available_agents]

        @tool
        def route_to_node(next_node: str) -> str:
            """Route to the next node in the workflow."""
            if next_node not in valid_targets:
                return f"Error: '{next_node}' is not a valid routing target."
            return f"Routing to {next_node}"

        llm_with_tools = llm.bind_tools([route_to_node], tool_choice="required")

        prompt_messages = supervisor_prompt.format_messages(
            available_agents=available_agents,
            messages=messages,
            workflow_plan=workflow_plan,
            agent_results=agent_results,
            valid_routing_targets=valid_targets
        )
        instruction = SystemMessage(
            content=f"You MUST call the 'route_to_node' tool with 'next_node' set to one of: {', '.join(valid_targets)}."
        )

        response = llm_with_tools.invoke([instruction] + prompt_messages)
        next_action = self._parse_routing_decision(response, valid_targets)

        return {"messages": [response], "next": next_action, "task_context": task_context}

    def _parse_routing_decision(self, response, valid_targets: List[str]) -> str:
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_call = response.tool_calls[0]
            next_node = tool_call.get('args', {}).get('next_node', '')
            if next_node in valid_targets:
                return next_node
            return "planner"
        if hasattr(response, 'content') and response.content:
            content = response.content.lower()
            for target in valid_targets:
                if target in content:
                    return target
        return "end"

    def _route_from_supervisor(self, state: OrchestratorState) -> str:
        next_node = state.get("next", "end")
        if next_node == "end":
            return "end"
        if not state.get("workflow_plan") and next_node not in ["planner", "end"]:
            return "planner"
        if state.get("workflow_plan") and self._all_tasks_completed(state):
            return "aggregator"
        return next_node

    def _planner_node(self, state: OrchestratorState) -> Dict[str, Any]:
        llm = model_registry.get_model(self.supervisor_model)
        planner_prompt = self.prompt_loader.load_prompt("planner")  # 修改：使用实例加载器

        messages = state["messages"]
        available_agents = [
            {"name": name, "capabilities": agent_registry.get_metadata(name).capabilities}
            for name in agent_registry.list_agents()
        ]

        response = llm.invoke(
            planner_prompt.format_messages(
                user_request=messages[-1].content if messages else "",
                available_agents=available_agents
            )
        )
        workflow_plan = self._parse_plan(response.content)
        return {"messages": [response], "workflow_plan": workflow_plan}

    def _executor_node(self, state: OrchestratorState) -> Dict[str, Any]:
        workflow_plan = state.get("workflow_plan", [])
        pending_tasks = [t for t in workflow_plan if t.get("status") != "completed"]
        if not pending_tasks:
            return {"messages": [AIMessage(content="All tasks completed")]}

        results = {}
        for task in pending_tasks:
            agent_name = task.get("agent")
            if agent_name and agent_name in agent_registry.list_agents():
                try:
                    agent_instance = agent_registry.get_agent(agent_name)
                    print(f"🚀 执行 Agent: {agent_name}")
                    # 传递原始用户消息和上下文
                    agent_response = agent_instance.invoke({
                        "messages": state.get("messages", []),
                        "task_context": state.get("task_context", {})
                    })
                    extracted = self._extract_agent_answer(agent_response)
                    results[agent_name] = extracted
                    print(f"✅ Agent {agent_name} 完成，结果长度: {len(extracted)}")
                except Exception as e:
                    results[agent_name] = f"执行错误: {str(e)}"
            else:
                results[task.get("id", "unknown")] = f"未指定代理 (期望: {agent_name})"

        return {
            "agent_results": results,
            "workflow_plan": self._update_plan_status(workflow_plan, results)
        }

    def _extract_agent_answer(self, agent_response: Any) -> str:
        if isinstance(agent_response, dict) and "messages" in agent_response:
            messages = agent_response["messages"]
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                    return msg.content
            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and msg.content:
                    return msg.content
            return str(messages)
        return str(agent_response)

    def _create_agent_node(self, agent_name: str):
        """Create Worker Agent node (修复消息历史问题)"""
        def agent_node(state: OrchestratorState) -> Dict[str, Any]:
            agent_instance = agent_registry.get_agent(agent_name)
            print(f"🎯 直接调用 Agent: {agent_name}")

            # --- 修复：清理消息历史，只保留 HumanMessage ---
            all_messages = state["messages"]
            cleaned_messages = [
                msg for msg in all_messages 
                if isinstance(msg, HumanMessage)
            ]
            task_context = state.get("task_context", {})
            if task_context.get("query") and not cleaned_messages:
                cleaned_messages = [HumanMessage(content=task_context["query"])]

            agent_response = agent_instance.invoke({
                "messages": cleaned_messages,
                "task_context": task_context
            })
            # --- 修复结束 ---

            extracted = self._extract_agent_answer(agent_response)
            agent_results = state.get("agent_results", {})
            agent_results[agent_name] = extracted
            return {
                "messages": [AIMessage(content=extracted)],
                "agent_results": agent_results
            }
        return agent_node

    def _aggregator_node(self, state: OrchestratorState) -> Dict[str, Any]:
        llm = model_registry.get_model(self.supervisor_model)
        aggregator_prompt = self.prompt_loader.load_prompt("aggregator")  # 修改：使用实例加载器
        agent_results = state.get("agent_results", {})
        user_request = state["messages"][0].content if state["messages"] else ""
        response = llm.invoke(
            aggregator_prompt.format_messages(user_request=user_request, agent_results=agent_results)
        )
        return {"messages": [response]}

    def _should_end(self, state: OrchestratorState) -> str:
        if state.get("error_count", 0) < self.max_retries:
            return "supervisor"
        return "end"

    def _all_tasks_completed(self, state: OrchestratorState) -> bool:
        plan = state.get("workflow_plan")
        if not plan:
            return False
        return all(t.get("status") == "completed" for t in plan)

    def _parse_plan(self, response_content: str) -> List[Dict[str, Any]]:
        try:
            content = response_content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            plan = json.loads(content)
            if isinstance(plan, list):
                return plan
            if isinstance(plan, dict) and "tasks" in plan:
                return plan["tasks"]
        except Exception:
            pass

        agents = agent_registry.list_agents()
        agent_name = agents[0] if agents else "web_research_agent"
        return [{
            "id": "task_1",
            "name": "处理用户请求",
            "agent": agent_name,
            "tools": [],
            "depends_on": [],
            "parallel": False,
            "status": "pending"
        }]

    def _update_plan_status(self, plan: List, results: Dict) -> List:
        for task in plan:
            if task.get("agent") in results:
                task["status"] = "completed"
        return plan

    def invoke(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "task_context": context or {},
            "agent_results": {},
            "workflow_plan": None,
            "error_count": 0,
            "max_retries": self.max_retries
        }
        return self.graph.invoke(initial_state)

    async def ainvoke(self, user_input: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        initial_state = {
            "messages": [HumanMessage(content=user_input)],
            "task_context": context or {},
            "agent_results": {},
            "workflow_plan": None,
            "error_count": 0,
            "max_retries": self.max_retries
        }
        return await self.graph.ainvoke(initial_state)