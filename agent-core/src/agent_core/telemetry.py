import logging
import os

from opentelemetry import trace
from opentelemetry._logs import set_logger_provider
from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

try:
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    _httpx_instrumentation_available = True
except ImportError:
    # httpx not yet installed; will be available once the Anthropic SDK is added
    _httpx_instrumentation_available = False

_DEFAULT_ENDPOINT = "otel-collector.otel.svc.cluster.local:4317"

# Span attribute names following OTel GenAI and MCP semantic conventions (semconv 1.41.0).
# Apply via span.set_attribute() in tool and agent span implementations.

# OTel GenAI spec attributes
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_RESPONSE_MODEL = "gen_ai.response.model"
GEN_AI_PROVIDER_NAME = "gen_ai.provider.name"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_REQUEST_MAX_TOKENS = "gen_ai.request.max_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"

# OTel MCP spec attributes
MCP_METHOD_NAME = "mcp.method.name"
MCP_SESSION_ID = "mcp.session.id"

# Custom extensions — no spec equivalent; namespaced to avoid future spec collisions
GENAI_PROMPT_VERSION = "genai.prompt.version"
LLM_TOOL_CALLS_MADE = "llm.tool_calls_made"
AGENT_SUBAGENTS_INVOKED = "agent.subagents_invoked"
AGENT_DELEGATION_REASON = "agent.delegation_reason"
EVAL_SCORE = "eval.score"

# Backwards-compat aliases — remove after one release cycle once dashboards are updated
LLM_MODEL = GEN_AI_REQUEST_MODEL
LLM_PROMPT_VERSION = GENAI_PROMPT_VERSION
LLM_INPUT_TOKENS = GEN_AI_USAGE_INPUT_TOKENS
LLM_OUTPUT_TOKENS = GEN_AI_USAGE_OUTPUT_TOKENS
MCP_TOOL_NAME = GEN_AI_TOOL_NAME


def setup_telemetry(service_name: str) -> None:
    """Initialise TracerProvider and LoggerProvider, bridge stdlib logging to OTEL.

    Never raises — if the collector is unreachable the app continues without telemetry.
    IMPORTANT: never pass Key Vault secret values as span attributes or log fields.
    """
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(levelname)s %(name)s %(message)s",
    )

    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", _DEFAULT_ENDPOINT)
    resource = Resource.create({"service.name": service_name})

    try:
        _setup_traces(endpoint, resource)
        _setup_logs(endpoint, resource)

        LoggingInstrumentor().instrument(set_logging_format=True)

        if _httpx_instrumentation_available:
            HTTPXClientInstrumentor().instrument()

        logging.info("event=telemetry_initialized service=%s endpoint=%s", service_name, endpoint)
    except Exception:
        logging.warning("event=telemetry_setup_failed service=%s endpoint=%s", service_name, endpoint, exc_info=True)


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def _setup_traces(endpoint: str, resource: Resource) -> None:
    exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


def _setup_logs(endpoint: str, resource: Resource) -> None:
    exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
    provider = LoggerProvider(resource=resource)
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
    set_logger_provider(provider)
