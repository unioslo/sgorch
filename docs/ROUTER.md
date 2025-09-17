# SGOrch Router Service

The SGOrch router is a lightweight FastAPI service that keeps a single, stable
endpoint in front of a pool of backend workers. It exists to give clients (for
example Hugging Face TEI deployments) a single URL that automatically load
balances, retries, and tracks worker health without asking the inference
backend to run its own router.

## Responsibilities

- Track the set of active workers via the `sgorch run` orchestrator, or manual
  `add/remove` calls.
- Watch worker health using periodic HTTP probes (default `GET /health`).
- Randomly load balance requests across healthy workers and retry on failure.
- Fail fast (HTTP 503) when no healthy workers are available.
- Maintain compatibility with the existing router REST contract so the current
  `RouterClient` implementation in SGOrch continues to work unchanged.

## Running the Router

Launch the router with the dedicated CLI command:

```bash
sgorch router \
  --host 0.0.0.0 \
  --port 8080 \
  --health-path /health \
  --probe-interval 10 \
  --probe-timeout 5 \
  --request-timeout 30 \
  --max-retries 3 \
  --failure-cooldown 5
```

### CLI Options

| Flag | Description |
| ---- | ----------- |
| `--host` | Interface to bind for client traffic (default `0.0.0.0`). |
| `--port`, `-p` | Listening port (default `8080`). |
| `--health-path` | Path the router probes on each worker (default `/health`). |
| `--probe-interval` | Seconds between health probes. |
| `--probe-timeout` | Timeout for each probe request. |
| `--request-timeout` | End-to-end timeout for proxied client requests. |
| `--max-retries` | Maximum upstream attempts per incoming request. |
| `--failure-cooldown` | Seconds to wait before re-probing an unhealthy worker. |
| `--log-level`, `-l` | Router log verbosity. |

The router is stateless: restart it at any time and SGOrch will repopulate
worker registrations via the REST API.

## REST API Contract

The router exposes the same HTTP contract the legacy SGLang router used, so the
existing `RouterClient` continues to work without modifications:

| Method | Endpoint | Description |
| ------ | -------- | ----------- |
| `GET` | `/workers/list` | Returns `{"workers": ["http://host:port", ...]}`. |
| `POST` | `/workers/add?url=<worker>` | Registers a worker; returns `{"status": "ok"}`. |
| `POST` | `/workers/remove?url=<worker>` | Deregisters a worker; 404 is treated as success. |

Any request that does not match the endpoints above is proxied to a healthy
worker. On each attempt the router:

1. Chooses a healthy worker at random (falling back to `unknown` status if no
   healthy workers exist yet).
2. Forwards the incoming HTTP method, body, and headers (excluding hop-by-hop
   headers), and appends `X-Forwarded-For` with the client address.
3. Retries with a different worker when the upstream request times out, fails,
   or returns a `5xx` response, up to `--max-retries` attempts.

## Health Monitoring

- Probes use the configured `--health-path` and treat any response with status
  code `< 500` as healthy.
- Repeated failures mark a worker as unhealthy and apply the cooldown before it
  is retried.
- Successful responses immediately move the worker back into the healthy pool.

## Integration with SGOrch

- SGLang deployments continue to register with their existing routers.
- TEI deployments (or any backend without a native router) can now opt into
  SGOrch's router, giving clients a single host/port to target.
- The orchestrator talks to the router via the `RouterClient`, so configuration
  stays the same (`router.base_url`, optional auth headers).
- Because the API contract is unchanged, the router can also be used for manual
  worker management or during migrations from the SGLang router.

## Operational Tips

- Use `curl http://<router-host>:<port>/healthz` for a lightweight readiness
  check. It returns the current worker set.
- Monitor logs for `Worker marked unhealthy` messages; they include reasons for
  removal (timeout, upstream `5xx`, etc.).
- When deploying alongside `sgorch run`, start the router first so the
  orchestrator can register workers immediately on boot.
- For Prometheus metrics, scrape the orchestrator; router-specific metrics can
  be added later using the in-process `prometheus_client` if needed.

