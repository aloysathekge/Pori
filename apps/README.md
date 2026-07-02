# apps — frontend surfaces

Deployable frontend surfaces (web, desktop, …). Each talks to a product backend
(e.g. `products/aloy/backend/`) over **REST + SSE** — never by Python import, so
these are **not** part of the `products → extensions → pori` import graph.

- `web/` — the web surface ← `pori_cloud_client`
- `desktop/` — the desktop shell (later)
