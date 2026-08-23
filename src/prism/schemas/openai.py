"""OpenAI Chat Completions wire types — the client contract.

Deliberately hand-written rather than imported from the OpenAI SDK: this is the
schema Prism *promises*, and it doubles as request validation. Fields Prism does
not support yet are still modelled, so they can be rejected with a precise error
instead of being silently ignored.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "developer", "user", "assistant", "tool"]


class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class ImageURL(BaseModel):
    url: str
    detail: str | None = None


class ImagePart(BaseModel):
    type: Literal["image_url"]
    image_url: ImageURL


ContentPart = TextPart | ImagePart


class FunctionCall(BaseModel):
    name: str
    # OpenAI serializes tool arguments as a JSON *string*; Anthropic uses a JSON
    # object. Translating between them means parse/serialize, not passthrough.
    arguments: str


class ToolCall(BaseModel):
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Role
    content: str | list[ContentPart] | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class FunctionDef(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[str, Any] | None = None
    strict: bool | None = None


class ToolDef(BaseModel):
    type: Literal["function"] = "function"
    function: FunctionDef


class NamedToolChoiceFunction(BaseModel):
    name: str


class NamedToolChoice(BaseModel):
    type: Literal["function"] = "function"
    function: NamedToolChoiceFunction


ToolChoice = Literal["none", "auto", "required"] | NamedToolChoice


class JSONSchemaSpec(BaseModel):
    name: str
    description: str | None = None
    schema_: dict[str, Any] | None = Field(default=None, alias="schema")
    strict: bool | None = None


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object", "json_schema"]
    json_schema: JSONSchemaSpec | None = None


class StreamOptions(BaseModel):
    include_usage: bool = False


class PrismOptions(BaseModel):
    """Prism-specific request extensions.

    Namespaced under a single key so a client that sends them to OpenAI proper
    gets one predictable error rather than several.
    """

    model_config = ConfigDict(extra="forbid")

    prompt_version: str | None = None
    # Reserved for later phases; accepted and recorded now so traces from week 2
    # remain comparable with traces from week 9.
    effort: Literal["low", "medium", "high", "xhigh", "max"] | None = None
    thinking: bool | None = None
    cache: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    model: str
    messages: list[ChatMessage]

    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    stream: bool = False
    stream_options: StreamOptions | None = None
    n: int | None = None

    tools: list[ToolDef] | None = None
    tool_choice: ToolChoice | None = None
    parallel_tool_calls: bool | None = None

    response_format: ResponseFormat | None = None
    user: str | None = None
    metadata: dict[str, Any] | None = None

    prism: PrismOptions | None = None

    @property
    def effective_max_tokens(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


# --- Response types -------------------------------------------------------


class PromptTokensDetails(BaseModel):
    cached_tokens: int = 0


class CompletionTokensDetails(BaseModel):
    reasoning_tokens: int = 0


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: PromptTokensDetails | None = None
    completion_tokens_details: CompletionTokensDetails | None = None
    # Non-standard, Prism-specific: the OpenAI usage object has nowhere to put
    # cache *writes*, and dropping them would make the trace uncostable.
    prism_cache_creation_input_tokens: int | None = None


class ResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    # Non-standard but widely adopted by OpenAI-compatible servers for reasoning
    # models. Only populated when the upstream returned summarized thinking.
    reasoning_content: str | None = None


FinishReason = Literal["stop", "length", "tool_calls", "content_filter"]


class Choice(BaseModel):
    index: int = 0
    message: ResponseMessage
    finish_reason: FinishReason | None = None
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: str | None = None
