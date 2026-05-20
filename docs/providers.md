# Providers

LangDeep ships provider registrations for common model families and a local
`mock` provider for examples and tests.

## Built-In Names

- `mock`
- `openai`
- `azure_openai`
- `deepseek`
- `anthropic`
- `google_genai`
- `vertexai`
- `ollama`

Optional provider dependencies are installed through extras such as
`.[anthropic]`, `.[google-genai]`, `.[vertexai]`, and `.[ollama]`.

## Configuration

```python
from langdeep import model


@model(name="gpt4o", provider="openai", model_name="gpt-4o")
def gpt4o():
    pass
```

Do not hard-code API keys in decorators. Prefer environment variables or
`SecretsManager`.
