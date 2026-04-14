import hashlib
import json
from pathlib import Path
from typing import Any

import litellm
from litellm import CustomStreamWrapper, Message
from litellm.exceptions import (
    AuthenticationError as LiteLLMAuthenticationError,
)
from litellm.exceptions import (
    ContextWindowExceededError as LiteLLMContextWindowExceededError,
)
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)
from litellm.utils import token_counter
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from libs.terminus_agent.llms.base_llm import (
    BaseLLM,
    ContextLengthExceededError,
    OutputLengthExceededError,
)
from libs.terminus_agent.utils.anthropic_caching import add_anthropic_caching
from libs.terminus_agent.utils.logger import logger


class LiteLLM(BaseLLM):
    PROMPT_TEMPLATE_PATH = (
        Path(__file__).parent / "prompt-templates/formatted-response.txt"
    )

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.7,
        api_base: str | None = None,
        api_key: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._model_name = model_name
        self._temperature = temperature
        self._supported_params = get_supported_openai_params(model_name)
        self._api_base = api_base
        self._api_key = api_key

        if self._supported_params is not None:
            self._supports_response_format = "response_format" in self._supported_params
            self._supports_temperature = "temperature" in self._supported_params
        else:
            self._supports_response_format = False
            self._supports_temperature = False

        if not self._supports_temperature:
            logger.warning(
                f"Model {self._model_name} does not support temperature. temperature={temperature} will be ignored."
            )

        self._prompt_template = self.PROMPT_TEMPLATE_PATH.read_text()

    def _clean_value(self, value):
        match value:
            case _ if callable(value):
                return None
            case dict():
                return {
                    k: v
                    for k, v in {
                        k: self._clean_value(v) for k, v in value.items()
                    }.items()
                    if v is not None
                }
            case list():
                return [
                    self._clean_value(v)
                    for v in value
                    if self._clean_value(v) is not None
                ]
            case str() | int() | float() | bool():
                return value
            case _:
                return str(value)

    def _redact_sensitive_fields(self, value: Any) -> Any:
        sensitive_keys = {"api_key", "x-api-key", "authorization"}

        if isinstance(value, dict):
            redacted: dict[str, Any] = {}
            for key, item in value.items():
                normalized_key = key.lower() if isinstance(key, str) else None
                if normalized_key in sensitive_keys:
                    item_str = item if isinstance(item, str) else str(item)
                    redacted[f"{key}_sha256"] = hashlib.sha256(item_str.encode()).hexdigest()
                    continue
                redacted[key] = self._redact_sensitive_fields(item)
            return redacted

        if isinstance(value, list):
            return [self._redact_sensitive_fields(item) for item in value]

        return value

    def _init_logger_fn(self, logging_path: Path):
        def logger_fn(model_call_dict: dict):
            clean_dict = self._clean_value(model_call_dict)
            clean_dict = self._redact_sensitive_fields(clean_dict)
            logging_path.write_text(
                json.dumps(
                    clean_dict,
                    indent=4,
                )
            )

        return logger_fn

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=15),
        retry=retry_if_not_exception_type(
            (
                ContextLengthExceededError,
                OutputLengthExceededError,
                LiteLLMAuthenticationError,
            )
        ),
    )
    def call(
        self,
        prompt: str,
        message_history: list[dict[str, Any] | Message] = [],
        response_format: dict | type[BaseModel] | None = None,
        logging_path: Path | None = None,
        **kwargs,
    ) -> str:
        if response_format is not None and not self._supports_response_format:
            if isinstance(response_format, dict):
                schema = json.dumps(response_format, indent=2)
            else:
                schema = json.dumps(response_format.model_json_schema(), indent=2)
            prompt = self._prompt_template.format(schema=schema, prompt=prompt)
            response_format = None

        logger.debug(f"Making call to {self._model_name}")

        if logging_path is not None:
            logger_fn = self._init_logger_fn(logging_path)
        else:
            logger_fn = None

        # Prepare messages with caching for Anthropic models
        messages = message_history + [{"role": "user", "content": prompt}]
        messages = add_anthropic_caching(messages, self._model_name)

        try:
            response = litellm.completion(
                model=self._model_name,
                messages=messages,
                temperature=self._temperature,
                response_format=response_format,
                drop_params=True,
                logger_fn=logger_fn,
                api_base=self._api_base,
                api_key=self._api_key,
                **kwargs,
            )
            logger.debug(f"LiteLLM response type: {type(response)}, response: {response}")
        except Exception as e:
            # Return the terminal-bench exception
            if isinstance(e, LiteLLMContextWindowExceededError):
                raise ContextLengthExceededError
            if isinstance(e, LiteLLMAuthenticationError):
                raise e  # Re-raise as-is so QualityChecker can catch it
            raise e

        if response is None:
            raise ValueError(
                f"LiteLLM returned None for model {self._model_name}. "
                f"Check vLLM server status, API base: {self._api_base}"
            )

        if isinstance(response, CustomStreamWrapper):
            raise NotImplementedError("Streaming is not supported for T bench yet")

        choices = response.get("choices")
        if not choices:
            raise ValueError(
                f"LiteLLM response has no choices. Response: {response}"
            )

        first_choice = choices[0]
        if hasattr(first_choice, "get"):
            finish_reason = first_choice.get("finish_reason")
            message = first_choice.get("message")
        else:
            finish_reason = getattr(first_choice, "finish_reason", None)
            message = getattr(first_choice, "message", None)

        if message is None:
            raise ValueError(
                f"LiteLLM response missing message in first choice. Response: {response}"
            )

        if hasattr(message, "get"):
            content = message.get("content")
        else:
            content = getattr(message, "content", None)

        # Sometimes the LLM returns a response with a finish reason of "length"
        # This typically means we hit the max_tokens limit, not the context window
        if finish_reason == "length":
            # Create exception with truncated response attached
            exc = OutputLengthExceededError(
                f"Model {self._model_name} hit max_tokens limit. Response was truncated. Consider increasing max_tokens if possible.",
                truncated_response=content if isinstance(content, str) else "",
            )
            raise exc

        if content is None:
            raise ValueError(
                f"LiteLLM response content is None. Response: {response}"
            )

        if not isinstance(content, str):
            raise ValueError(
                "LiteLLM response content is not a string. "
                f"Type: {type(content)}, response: {response}"
            )

        if not content.strip():
            raise ValueError(
                f"LiteLLM response content is empty. Response: {response}"
            )

        return content

    def count_tokens(self, messages: list[dict]) -> int:
        return token_counter(model=self._model_name, messages=messages)
