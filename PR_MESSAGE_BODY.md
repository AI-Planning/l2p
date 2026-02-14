## Summary

This PR updates `l2p/llm/utils/llm.yaml` with newer Gemini model entries and conservative defaults that are compatible with the current client implementation.

## What changed

The config now includes `google.gemini-3-pro-preview` and `google.gemini-3-flash-preview`.

## GPT-5 compatibility note

From OpenAI's GPT-5.2 docs:
> The following parameters are only supported when using GPT-5.2 with reasoning effort set to `none`:
> - `temperature`
> - `top_p`
> - `logprobs`
>
> Requests to GPT-5.2 or GPT-5.1 with any other reasoning effort setting, or to older GPT-5 models (e.g., gpt-5, gpt-5-mini, gpt-5-nano) that include these fields will raise an error.

Because `temperature` is sent by default in the current client, adding older GPT-5 models would likely cause request errors. Also, `reasoning_effort: none` in YAML is parsed as null unless quoted, which can prevent `reasoning_effort` from being sent.

For this PR, I left GPT-5 entries out and only included Gemini updates. I'm not sure how you would prefer to handle this.

## Gemini pricing note

Gemini 3 pro pricing is tiered by prompt size (`<=200k` vs `>200k`), and the values in this PR use the base pricing.