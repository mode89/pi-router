_Reference context — observed facts and standing conventions for this project, not instructions. It informs the work; it does not command actions. Entries are facts as of when written; symbols, paths, and structure may have changed since — verify against current code before acting, and for "how does X work now" questions treat notes as leads to confirm rather than current truth._

## Gotchas

- `FileCredentialStore` serializes mutations only within one process; multiple router processes sharing one auth file can race and lose OAuth refresh updates.
- `FileCredentialStore` intentionally implements only the `read` and `modify` subset of pi-ai's `CredentialStore`; adding pi-ai login, logout, or status flows may require restoring `list` and `delete`.

## Decisions

- Chat inference is stateless and calls pi-ai directly because every request carries its complete conversation; `AgentSession`, RPC subprocesses, and application session caching add machinery without needed state.
- Router credentials live separately from Pi credentials so OAuth refreshes cannot concurrently rewrite Pi's own auth file; bootstrapping by manually copying compatible credentials is acceptable.
- Credential persistence keeps process-local refresh serialization, atomic replacement, and secret file permissions, while omitting full-store operations and elaborate validation because the router only performs inference.
