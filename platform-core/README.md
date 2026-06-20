# platform-core

Shared infra plumbing for the `claudeaiportfolio` services — one source of truth
instead of copy-pasting boilerplate per repo (centralise-don't-copy). Consumed as
a git-subdirectory dependency, like `agent-evals`.

Today it ships the **`queue`** module (RQ-on-Redis). Other modules (db, llm,
embeddings, otel, chunking) are pulled in lazily as services need them, each
behind its own optional extra.

## Install

```toml
# in a consumer's pyproject.toml
dependencies = [
  "platform-core[queue] @ git+https://github.com/claudeaiportfolio/ai-infra-templates.git@platform-core-v0.1.0#subdirectory=platform-core",
]
```

## `queue` — RQ on Redis

A TLS-capable Redis connection factory + RQ `Queue`/`Worker` builders. Secrets
come from the environment (a Key Vault-synced k8s Secret); ACL users scope each
consumer to least privilege; `rediss://` / `use_tls=True` enables in-cluster TLS.

```python
from platform_core.queue import RedisSettings, redis_connection, get_queue, build_worker, Retry

settings = RedisSettings.from_url(os.environ["REDIS_URL"], password=os.environ["REDIS_PASSWORD"])
conn = redis_connection(settings)

# producer
get_queue("embed-jobs", conn).enqueue(
    "embedding_worker.tasks.process_document", payload, retry=Retry(max=3, interval=[10, 30, 60])
)

# worker
build_worker(["embed-jobs"], conn).work()
```

## Versioning

Tagged `platform-core-vX.Y.Z`; consumers pin to a tag. Tests run in CI on changes
under `platform-core/**`.
