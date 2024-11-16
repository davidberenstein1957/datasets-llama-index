from datasets_llama_index.datasets_handler import DatasetsHandler
from llama_index.core import Document, VectorStoreIndex
from llama_index.core.instrumentation import get_dispatcher

# root dispatcher
root_dispatcher = get_dispatcher()

# register span handler
event_handler = DatasetsHandler(repo_id="llama-index-test")
root_dispatcher.add_span_handler(event_handler)
root_dispatcher.add_event_handler(event_handler)
index = VectorStoreIndex.from_documents([Document.example()])

query_engine = index.as_query_engine()

query_engine.query("Tell me about LLMs?")
