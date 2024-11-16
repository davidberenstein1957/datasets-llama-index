# Copyright 2024-present, Argilla, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import atexit
import inspect
import json
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from huggingface_hub import CommitScheduler
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.events.agent import (
    AgentChatWithStepEndEvent,
    AgentChatWithStepStartEvent,
    AgentRunStepEndEvent,
    AgentRunStepStartEvent,
    AgentToolCallEvent,
)
from llama_index.core.instrumentation.events.chat_engine import (
    StreamChatDeltaReceivedEvent,
    StreamChatErrorEvent,
)
from llama_index.core.instrumentation.events.embedding import (
    EmbeddingEndEvent,
    EmbeddingStartEvent,
)
from llama_index.core.instrumentation.events.llm import (
    LLMChatEndEvent,
    LLMChatInProgressEvent,
    LLMChatStartEvent,
    LLMCompletionEndEvent,
    LLMCompletionStartEvent,
    LLMPredictEndEvent,
    LLMPredictStartEvent,
    LLMStructuredPredictEndEvent,
    LLMStructuredPredictStartEvent,
)
from llama_index.core.instrumentation.events.query import (
    QueryEndEvent,
    QueryStartEvent,
)
from llama_index.core.instrumentation.events.rerank import (
    ReRankEndEvent,
    ReRankStartEvent,
)
from llama_index.core.instrumentation.events.retrieval import (
    RetrievalEndEvent,
    RetrievalStartEvent,
)
from llama_index.core.instrumentation.events.span import (
    SpanDropEvent,
)
from llama_index.core.instrumentation.events.synthesis import (
    GetResponseStartEvent,
    SynthesizeEndEvent,
    SynthesizeStartEvent,
)
from llama_index.core.instrumentation.span.simple import SimpleSpan
from llama_index.core.instrumentation.span_handlers import (
    SimpleSpanHandler,
)

context_root: ContextVar[Union[Tuple[str, str], Tuple[None, None]]] = ContextVar(
    "context_root", default=(None, None)
)


if TYPE_CHECKING:
    from huggingface_hub import HfApi


class DatasetsHandler(SimpleSpanHandler, BaseEventHandler, extra="allow"):
    """
    Handler that logs predictions to Datasets.

    This handler automatically logs the predictions made with LlamaIndex to Datasets,
    without the need to create a dataset and log the predictions manually. Events relevant
    to the predictions are automatically logged to Datasets as well, including timestamps of
    all the different steps of the retrieval and prediction process.

    Attributes:


    Usage:
        ```python
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
        from llama_index.core.query_engine import RetrieverQueryEngine
        from llama_index.core.instrumentation import get_dispatcher
        from llama_index.core.retrievers import VectorIndexRetriever
        from llama_index.llms.openai import OpenAI

        from datasets_llama_index import DatasetsHandler

        datasets_handler = DatasetsHandler(

        )
        root_dispatcher = get_dispatcher()
        root_dispatcher.add_span_handler(datasets_handler)
        root_dispatcher.add_event_handler(datasets_handler)

        Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.8, openai_api_key=os.getenv("OPENAI_API_KEY"))

        documents = SimpleDirectoryReader("../../data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine(similarity_top_k=2)

        response = query_engine.query("What did the author do growing up?")
        ```
    """

    events: List[Dict[str, Any]] = []
    spans: Dict[str, SimpleSpan] = {}

    def __init__(
        self,
        *,
        repo_id: str,
        folder_path: Optional[Union[str, Path]] = None,
        every: Union[int, float] = 5,
        path_in_repo: Optional[str] = None,
        revision: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        allow_patterns: Optional[Union[List[str], str]] = None,
        ignore_patterns: Optional[Union[List[str], str]] = None,
        squash_history: bool = False,
        hf_api: Optional["HfApi"] = None,
    ) -> None:
        super().__init__()

        self.folder_path = Path("traces") if folder_path is None else folder_path
        device_id = uuid.uuid4()
        self.feedback_file = self.folder_path / f"data_{device_id}.json"

        self.scheduler = CommitScheduler(
            repo_id=repo_id,
            folder_path=self.folder_path,
            every=every,
            path_in_repo=path_in_repo,
            repo_type="dataset",
            revision=revision,
            private=private,
            token=token,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            squash_history=squash_history,
            hf_api=hf_api,
        )

        atexit.register(self.scheduler.push_to_hub)

    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "DatasetsHandler"

    def get_events_by_span_id(self, span_id: str) -> List[Dict[str, Any]]:
        return [event for event in self.events if event["event_span_id"] == span_id]

    def _replace_empty_dicts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        def _replace_empty_dicts_recursive(value: Any) -> Any:
            if isinstance(value, dict):
                if not value:
                    return None
                return {k: _replace_empty_dicts_recursive(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_replace_empty_dicts_recursive(item) for item in value]
            return value

        return _replace_empty_dicts_recursive(data)

    def handle(self, event: BaseEvent) -> None:
        """
        Logic to handle different events.

        Args:
            event (BaseEvent): The event to be handled.

        Returns:
            None
        """

        event_data = {
            "event_id": event.id_,
            "event_type": event.class_name(),
            "event_span_id": event.span_id,
            "event_timestamp": event.timestamp.timestamp(),
            "event_tags": event.tags if event.tags else None,
        }

        if isinstance(event, AgentRunStepStartEvent):
            event_data.update(
                {
                    "task_id": event.task_id,
                    "step": event.step,
                    "input": event.input,
                }
            )
        if isinstance(event, AgentRunStepEndEvent):
            event_data.update({"step_output": event.step_output})
        if isinstance(event, AgentChatWithStepStartEvent):
            event_data.update({"user_msg": event.user_msg})
        if isinstance(event, AgentChatWithStepEndEvent):
            event_data.update({"response": event.response})
        if isinstance(event, AgentToolCallEvent):
            event_data.update(
                {
                    "arguments": event.arguments,
                    "tool_name": event.tool.name,
                    "tool_description": event.tool.description,
                    "tool_openai_tool": event.tool.to_openai_tool(),
                }
            )
        if isinstance(event, StreamChatDeltaReceivedEvent):
            event_data.update({"delta": event.delta})
        if isinstance(event, StreamChatErrorEvent):
            event_data.update({"exception": event.exception})
        if isinstance(event, EmbeddingStartEvent):
            event_data.update({"model_dict": event.model_dict})
        if isinstance(event, EmbeddingEndEvent):
            event_data.update({"chunks": event.chunks, "embeddings": event.embeddings})
        if isinstance(event, LLMPredictStartEvent):
            event_data.update(
                {"template": event.template, "template_args": event.template_args}
            )
        if isinstance(event, LLMPredictEndEvent):
            event_data.update({"output": event.output})
        if isinstance(event, LLMStructuredPredictStartEvent):
            event_data.update(
                {
                    "template": event.template,
                    "template_args": event.template_args,
                    "output_cls": event.output_cls,
                }
            )
        if isinstance(event, LLMStructuredPredictEndEvent):
            event_data.update({"output": event.output})
        if isinstance(event, LLMCompletionStartEvent):
            event_data.update(
                {
                    "model_dict": event.model_dict,
                    "prompt": event.prompt,
                    "additional_kwargs": event.additional_kwargs,
                }
            )
        if isinstance(event, LLMCompletionEndEvent):
            event_data.update({"response": event.response, "prompt": event.prompt})
        if isinstance(event, LLMChatInProgressEvent):
            event_data.update({"messages": event.messages, "response": event.response})
        if isinstance(event, LLMChatStartEvent):
            event_data.update(
                {
                    "messages": event.messages,
                    "additional_kwargs": event.additional_kwargs,
                    "model_dict": event.model_dict,
                }
            )
        if isinstance(event, LLMChatEndEvent):
            event_data.update({"messages": event.messages, "response": event.response})
        if isinstance(event, RetrievalStartEvent):
            event_data.update({"str_or_query_bundle": event.str_or_query_bundle})
        if isinstance(event, RetrievalEndEvent):
            event_data.update(
                {"str_or_query_bundle": event.str_or_query_bundle, "nodes": event.nodes}
            )
        if isinstance(event, ReRankStartEvent):
            event_data.update(
                {
                    "query": event.query,
                    "nodes": event.nodes,
                    "top_n": event.top_n,
                    "model_name": event.model_name,
                }
            )
        if isinstance(event, ReRankEndEvent):
            event_data.update({"nodes": event.nodes})
        if isinstance(event, QueryStartEvent):
            event_data.update({"query": event.query})
        if isinstance(event, QueryEndEvent):
            event_data.update({"response": event.response, "query": event.query})
        if isinstance(event, SpanDropEvent):
            event_data.update({"err_str": event.err_str})
        if isinstance(event, SynthesizeStartEvent):
            event_data.update({"query": event.query})
        if isinstance(event, SynthesizeEndEvent):
            event_data.update({"response": event.response, "query": event.query})
        if isinstance(event, GetResponseStartEvent):
            event_data.update({"query_str": event.query_str})

        for key in ["response", "str_or_query_bundle", "query", "template"]:
            if key in event_data:
                event_data[key] = (
                    event_data[key].model_dump() if event_data[key] else None
                )

        for key, value in event_data.items():
            if not isinstance(value, (str, int, float, bool, type(None), list, dict)):
                try:
                    json.dumps(value)
                except (TypeError, OverflowError):
                    event_data[key] = str(value)
            try:
                json.dumps(value)
            except Exception:
                event_data[key] = str(value)
        event_data = self._replace_empty_dicts(event_data)

        self.events.append(event_data)

    def new_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        parent_span_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """
        Create a new span using the SimpleSpan class. If the span is the root span, it generates a new trace ID.

        Args:
            id_ (str): The unique identifier for the new span.
            bound_args (inspect.BoundArguments): The arguments that were bound to when the span was created.
            instance (Optional[Any], optional): The instance associated with the span, if present. Defaults to None.
            parent_span_id (Optional[str], optional): The identifier of the parent span. Defaults to None.
            tags (Optional[Dict[str, Any]], optional): Additional information about the span. Defaults to None.

        Returns:
            Optional[SimpleSpan]: The newly created SimpleSpan object if the span is successfully created.
        """
        trace_id, root_span_id = context_root.get()

        if not parent_span_id:
            trace_id = str(uuid.uuid4())
            root_span_id = id_
            context_root.set((trace_id, root_span_id))

        span = SimpleSpan(id_=id_, parent_id=parent_span_id, tags=tags or {})
        if id_ not in self.spans:
            self.spans[id_] = []
        self.spans[id_].append(span)
        return span

    def prepare_to_exit_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        result: Optional[Any] = None,
        **kwargs: Any,
    ) -> SimpleSpan:
        trace_id, root_span_id = context_root.get()
        if not trace_id:
            return None

        span: SimpleSpan = super().prepare_to_exit_span(
            id_, bound_args, instance, result, **kwargs
        )

        if id_ == root_span_id:
            self._save_to_commit_scheduler(trace_id=trace_id)
            self.spans.clear()
            self.events.clear()
            context_root.set((None, None))

        return span

    def prepare_to_drop_span(
        self,
        id_: str,
        bound_args: inspect.BoundArguments,
        instance: Optional[Any] = None,
        err: Optional[BaseException] = None,
        **kwargs: Any,
    ) -> Optional[SimpleSpan]:
        """Logic for droppping a span."""
        trace_id, root_span_id = context_root.get()
        span: SimpleSpan = super().prepare_to_drop_span(
            id_, bound_args, instance, err, **kwargs
        )
        if "workflow.run" in root_span_id.lower():
            self._save_to_commit_scheduler(trace_id=trace_id)
            self.spans.clear()
            self.events.clear()
            context_root.set((None, None))

        return span

    def _save_to_commit_scheduler(
        self,
        trace_id: str,
    ) -> None:
        try:
            with self.scheduler.lock:
                with self.feedback_file.open("a") as f:
                    for value in list(self.spans.values()):
                        for span in value:
                            data = {"trace_id": trace_id}
                            data.update(span.model_dump())
                            data["start_time"] = span.start_time.timestamp()
                            data["end_time"] = span.end_time.timestamp()

                            # unwrap events over rows
                            events = self.get_events_by_span_id(span.id_)
                            if events:
                                for event in events:
                                    row_data = data.copy()
                                    shared_event_columns = [
                                        "event_type",
                                        "event_timestamp",
                                        "event_tags",
                                        "event_span_id",
                                        "event_id",
                                    ]
                                    row_data.update(
                                        {
                                            k: v
                                            for k, v in event.items()
                                            if k in shared_event_columns
                                        }
                                    )
                                    # Copy all event keys except the standard ones we already handled
                                    event_metadata = {
                                        k: v
                                        for k, v in event.items()
                                        if k not in shared_event_columns
                                    }

                                    row_data["event_metadata"] = event_metadata

                                    row_data = self._replace_empty_dicts(row_data)

                                    f.write(json.dumps(row_data))
                                    f.write("\n")
                            else:
                                data = self._replace_empty_dicts(data)
                                json_data = json.dumps(obj=data)
                                f.write(json_data)
                                f.write("\n")
        except Exception as e:
            print(e)
            raise e
