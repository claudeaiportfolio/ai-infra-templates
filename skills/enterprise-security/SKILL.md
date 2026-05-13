---
name: enterprise-security
description: >
  Enforces production-grade security patterns for credential handling, authentication,
  and secrets management across claudeaiportfolio projects. Use this skill whenever
  working on GitHub Actions workflows, Azure infrastructure, Python application code,
  Snowflake connections, MCP server implementations, or any code that touches
  credentials, secrets, or authentication. Also triggers when reviewing code for
  security issues, designing CI/CD pipelines, or deciding how to pass configuration
  between services. If there's any chance credentials or auth are involved, use this skill.
---

# Enterprise Security Patterns

This skill defines the security baseline for all `claudeaiportfolio` projects.
These patterns simulate what a UK regulated financial services environment would
require. Apply them without being asked.

---

## Core principle

**Secrets never touch environment variables, workflow steps, or log output.**
The managed identity is the credential. Everything else flows from it.

---

## Authentication hierarchy

Always prefer in this order. Never suggest a pattern lower in the list when a
higher one is available.

| Pattern | Use when | Never use when |
|---|---|---|
| Workload identity federation (OIDC) | GitHub Actions → Azure | A service principal with a secret would work |
| Managed identity (`DefaultAzureCredential`) | Any Azure compute (AKS, runner) | A connection string would work |
| Key-pair authentication | Snowflake service accounts | Password auth would work |
| Certificate-based auth | Non-Azure services without managed identity support | Password auth would work |
| ❌ Service principal + client secret | Never in new code | — |
| ❌ Password authentication | Never in new code | — |
| ❌ Hardcoded credentials | Never | — |

---

## GitHub Actions — the correct pattern

### What the workflow does
- Authenticates to Azure via OIDC (`azure/login@v2`) using workload identity federation
- Passes only non-sensitive config as env vars (`KEY_VAULT_URL`, role names, warehouse names)
- Does NOT fetch secrets in shell steps
- Does NOT write secrets to `$GITHUB_ENV`

### What the application code does
- Fetches secrets from Key Vault at runtime using `DefaultAzureCredential`
- `DefaultAzureCredential` resolves automatically: OIDC token on runners, managed identity on AKS, CLI on local dev
- Same code works in all environments with no changes

### Correct pattern
```yaml
# workflow — passes only non-sensitive config
- name: Deploy
  run: uv run python deploy.py
  env:
    KEY_VAULT_URL: https://myvault.vault.azure.net/
    SNOWFLAKE_ROLE: MY_ROLE
```

```python
# application code — fetches secrets at runtime
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

credential = DefaultAzureCredential()
client = SecretClient(vault_url=os.environ["KEY_VAULT_URL"], credential=credential)
secret = client.get_secret("my-secret").value
```

### Anti-patterns to reject immediately
```yaml
# ❌ WRONG — secret touches $GITHUB_ENV
- run: |
    echo "MY_SECRET=$(az keyvault secret show --name x --query value -o tsv)" >> $GITHUB_ENV

# ❌ WRONG — even with add-mask, secret is in env var
- run: |
    SECRET=$(az keyvault secret show --name x --query value -o tsv)
    echo "::add-mask::$SECRET"
    echo "MY_SECRET=$SECRET" >> $GITHUB_ENV
```

If you see either pattern, flag as CRITICAL and replace with the runtime fetch pattern.

---

## Snowflake authentication

Always use key-pair authentication for service accounts. Never use passwords.

### Key-pair auth pattern
```python
from cryptography.hazmat.primitives.serialization import (
    load_pem_private_key, Encoding, PrivateFormat, NoEncryption
)

# Fetch private key PEM from Key Vault
private_key_pem = client.get_secret("snowflake-private-key").value

# Convert PEM → DER for the Snowflake connector
private_key = load_pem_private_key(private_key_pem.encode(), password=None)
private_key_der = private_key.private_bytes(
    encoding=Encoding.DER,
    format=PrivateFormat.PKCS8,
    encryption_algorithm=NoEncryption(),
)

conn = snowflake.connector.connect(
    account=account,
    user=user,
    private_key=private_key_der,
    role=role,
    warehouse=warehouse,
)
```

### RBAC pattern
- Service accounts use a dedicated read-only role (e.g., `FORECASTING_READER`)
- Role has minimum required grants — SELECT on specific tables, no DDL
- Schema deploy scripts run as `ACCOUNTADMIN` only in CI, never in application code
- Write operations (e.g., audit log INSERT) get a separate explicit grant

---

## Key Vault secret naming convention

Use kebab-case, service-prefixed:

```
snowflake-account
snowflake-user
snowflake-private-key
anthropic-api-key
auth0-client-id
auth0-client-secret
```

---

## Federated credential configuration

Subject identifier format for GitHub Actions:

```
repo:{org}/{repo}:ref:refs/heads/*     # all branches — portfolio dev
repo:{org}/{repo}:ref:refs/heads/main  # main only — production hardening
```

For pull request workflows:
```
repo:{org}/{repo}:pull_request
```

Never use environment-scoped subject identifiers unless the GitHub environment
is also configured on the federated credential.

---

## Logging hygiene

Never log values fetched from Key Vault, even at DEBUG level.

```python
# ✅ correct
logger.info("Connecting to Snowflake account: %s as user: %s", account, user)

# ❌ wrong — private key in logs
logger.debug("Using private key: %s", private_key_pem)

# ❌ wrong — connection string contains credentials
logger.debug("Connection config: %s", conn_config)
```

---

## Security review action

All `claudeaiportfolio` repos include the shared security review action:

```yaml
# .github/workflows/security-review.yml
- uses: claudeaiportfolio/claude-security-review/action@main
  with:
    anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
    github-token: ${{ secrets.GITHUB_TOKEN }}
```

This runs on every PR and flags violations of the patterns in this skill.
See `claudeaiportfolio/claude-security-review` for implementation details.

---

## Quick reference — decision tree

```
Need a credential in CI?
├── Is it an Azure resource?
│   └── Yes → azure/login OIDC + DefaultAzureCredential in app code
├── Is it Snowflake?
│   └── Yes → key-pair auth, private key in Key Vault
├── Is it a third-party API key?
│   └── Yes → store in Key Vault, fetch at runtime in app code
└── Am I about to write it to $GITHUB_ENV?
    └── Yes → STOP. Fetch it in the Python script instead.
```