## Summary

This PR updates `l2p/llm/utils/llm.yaml` with newer model entries and conservative defaults that are compatible with the current client implementation.

## What changed

The config now includes `openai.gpt-5.2`, `google.gemini-3-pro-preview`, and `google.gemini-3-flash-preview`. For GPT-5.2, this PR sets `reasoning_effort: none` to avoid compatibility issues.

## GPT-5 compatibility note

OpenAI docs state:
> GPT-5.2 parameter compatibility

> The following parameters are only supported when using GPT-5.2 with reasoning effort set to none:

>    temperature
>    top_p
>    logprobs

> Requests to GPT-5.2 or GPT-5.1 with any other reasoning effort setting, or to older GPT-5 models (e.g., gpt-5, gpt-5-mini, gpt-5-nano) that include these fields will raise an error.

This is incompatible with the current OpenAI client call pattern in this repo. So I did not include those models and only included 5.2 with reasoning set to none.

## Gemini pricing note

Gemini 3 Pro pricing is tiered by prompt size (`<=200k` vs `>200k`), but the current config schema stores only one `input` and one `output` value per model. Since this repo also uses conservative defaults (`context_length: 8192`, `max_completion_tokens: 4096`), normal usage does not typically reach the `>200k` tier. The values in this PR use the base pricing.

## Endpoint compatibility note

Gemini entries use the same parameter shape as the existing config for consistency. Native Gemini endpoints can support different parameter names/behavior depending on endpoint. 