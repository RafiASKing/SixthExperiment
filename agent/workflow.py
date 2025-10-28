"""Agent workflow helpers for the ticket booking LangGraph app."""

from __future__ import annotations

from typing import Any, Callable, MutableMapping

from langgraph.graph import StateGraph, END

StateDict = MutableMapping[str, Any]
NodeHandler = Callable[[StateDict], dict[str, Any]]


def compile_ticket_agent_workflow(
	*,
	state_type: type,
	router: Callable[[StateDict], str],
	classify_intent: NodeHandler,
	browsing_agent: NodeHandler,
	find_movie: NodeHandler,
	find_showtime: NodeHandler,
	select_seats: NodeHandler,
	confirm_booking: NodeHandler,
	execute_booking: NodeHandler,
	final_response: NodeHandler,
):
	"""Construct and compile the ticket agent workflow.

	The function wires the provided node callbacks into a :class:`StateGraph`
	with the expected routing logic for the cinema ticket assistant.
	"""

	workflow = StateGraph(state_type)
	workflow.add_node("classify_intent", classify_intent)
	workflow.add_node("browsing_agent", browsing_agent)
	workflow.add_node("find_movie", find_movie)
	workflow.add_node("find_showtime", find_showtime)
	workflow.add_node("select_seats", select_seats)
	workflow.add_node("confirm_booking", confirm_booking)
	workflow.add_node("execute_booking", execute_booking)
	workflow.add_node("final_response", final_response)

	workflow.set_entry_point("classify_intent")

	workflow.add_conditional_edges(
		"classify_intent",
		router,
		{
			"browsing_agent": "browsing_agent",
			"find_movie": "find_movie",
			"find_showtime": "find_showtime",
			"select_seats": "select_seats",
			"confirm_booking": "confirm_booking",
			"execute_booking": "execute_booking",
			"__end__": END,
		},
	)

	workflow.add_edge("browsing_agent", END)
	workflow.add_edge("find_movie", END)
	workflow.add_edge("find_showtime", END)
	workflow.add_edge("select_seats", END)
	workflow.add_edge("confirm_booking", END)
	workflow.add_edge("execute_booking", "final_response")
	workflow.add_edge("final_response", END)

	return workflow.compile()


__all__ = ["compile_ticket_agent_workflow"]
