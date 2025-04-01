"""
This calls an OpenAI-compatible server by VLLM.
"""

import os
from typing import Any, Dict, List, Optional, Union

from openai import APITimeoutError, ConflictError, InternalServerError, OpenAI
from openai import RateLimitError
from openai.types.chat import ChatCompletion
from pydantic import BaseModel

from modelgauge.prompt import ChatPrompt, ChatRole, TextPrompt
from modelgauge.retry_decorator import retry
from modelgauge.secret_values import InjectSecret, RequiredSecret, SecretDescription
from modelgauge.sut import PromptResponseSUT, SUTOptions, SUTResponse
from modelgauge.sut import TokenProbability, TopTokens
from modelgauge.sut_capabilities import AcceptsChatPrompt, AcceptsTextPrompt
from modelgauge.sut_capabilities import ProducesPerTokenLogProbabilities
from modelgauge.sut_decorator import modelgauge_sut
from modelgauge.sut_registry import SUTS


class OpenAIChatMessage(BaseModel):
    content: str
    role: str
    name: Optional[str] = None


_SYSTEM_ROLE = "system"
_USER_ROLE = "user"
_ASSISTANT_ROLE = "assistant"

_ROLE_MAP = {
    ChatRole.user: _USER_ROLE,
    ChatRole.sut: _ASSISTANT_ROLE,
    ChatRole.system: _SYSTEM_ROLE,
}


class VLLMBaseURL(RequiredSecret):
    @classmethod
    def description(cls) -> SecretDescription:
        return SecretDescription(scope="vllm", key="base_url", instructions="The base URL for querying models")


class OpenAIChatRequest(BaseModel):
    # https://platform.openai.com/docs/api-reference/chat/create
    messages: List[OpenAIChatMessage]
    model: str
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[bool] = None
    logprobs: Optional[bool] = None
    top_logprobs: Optional[int] = None
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    response_format: Optional[Dict] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    tools: Optional[List] = None
    tool_choice: Optional[Union[str, Dict]] = None
    user: Optional[str] = None


@modelgauge_sut(capabilities=[AcceptsTextPrompt, AcceptsChatPrompt, ProducesPerTokenLogProbabilities])
class VLLMOpenAIChat(PromptResponseSUT[OpenAIChatRequest, ChatCompletion]):
    """
    Documented at https://platform.openai.com/docs/api-reference/chat/create
    """

    def __init__(self, uid: str, model: str, base_url: str):
        super().__init__(uid)
        self.model = model
        self.base_url = base_url
        self.client: Optional[OpenAI] = None

    def _load_client(self) -> OpenAI:
        return OpenAI(base_url=self.base_url, max_retries=7)

    def translate_text_prompt(self, prompt: TextPrompt, options: SUTOptions) -> OpenAIChatRequest:
        messages = [OpenAIChatMessage(content=prompt.text, role=_USER_ROLE)]
        return self._translate_request(messages, options)

    def translate_chat_prompt(self, prompt: ChatPrompt, options: SUTOptions) -> OpenAIChatRequest:
        messages = []
        for message in prompt.messages:
            messages.append(OpenAIChatMessage(content=message.text, role=_ROLE_MAP[message.role]))
        return self._translate_request(messages, options)

    def _translate_request(self, messages: List[OpenAIChatMessage], options: SUTOptions):
        optional_kwargs: Dict[str, Any] = {}
        if options.top_logprobs is not None:
            optional_kwargs["logprobs"] = True
            optional_kwargs["top_logprobs"] = min(options.top_logprobs, 20)
        return OpenAIChatRequest(
            messages=messages,
            model=self.model,
            frequency_penalty=options.frequency_penalty,
            max_tokens=options.max_tokens,
            presence_penalty=options.presence_penalty,
            stop=options.stop_sequences,
            temperature=options.temperature,
            top_p=options.top_p,
            **optional_kwargs,
        )

    @retry(transient_exceptions=[APITimeoutError, ConflictError, InternalServerError, RateLimitError])
    def evaluate(self, request: OpenAIChatRequest) -> ChatCompletion:
        if self.client is None:
            # Handle lazy init.
            self.client = self._load_client()
        request_dict = request.model_dump(exclude_none=True)
        return self.client.chat.completions.create(**request_dict)

    def translate_response(self, request: OpenAIChatRequest, response: ChatCompletion) -> SUTResponse:
        assert len(response.choices) == 1, f"Expected a single response message, got {len(response.choices)}."
        choice = response.choices[0]
        text = choice.message.content
        logprobs: Optional[List[TopTokens]] = None
        if request.logprobs:
            logprobs = []
            assert (
                choice.logprobs is not None and choice.logprobs.content is not None
            ), "Expected logprobs, but not returned."
            for token_content in choice.logprobs.content:
                top_tokens: List[TokenProbability] = []
                for top in token_content.top_logprobs:
                    top_tokens.append(TokenProbability(token=top.token, logprob=top.logprob))
                logprobs.append(TopTokens(top_tokens=top_tokens))
        assert text is not None
        return SUTResponse(text=text, top_logprobs=logprobs)


SUTS.register(
    VLLMOpenAIChat,
    "OLMo-2-0325-32B-Instruct",
    "/weka/oe-adapt-default/ljm/models/allenai___OLMo-2-0325-32B-Instruct",
    os.environ.get("VLLM_HOST_BASEURL_OLMO2_32B", None),
)

SUTS.register(
    VLLMOpenAIChat,
    "OLMo-2-1124-13B-Instruct",
    "/weka/oe-adapt-default/ljm/models/allenai___OLMo-2-1124-13B-Instruct",
    os.environ.get("VLLM_HOST_BASEURL_OLMO2_13B", None),
)
