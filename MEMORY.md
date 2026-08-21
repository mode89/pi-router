_Reference context — observed facts and standing conventions for this project, not instructions. It informs the work; it does not command actions. Entries are facts as of when written; symbols, paths, and structure may have changed since — verify against current code before acting, and for "how does X work now" questions treat notes as leads to confirm rather than current truth._

## Conventions

- Tests (`npm test`) and lint (`npm run lint`) run without asking the user first. Why: both are read-only and fast. How to apply: after any code change, instead of proposing the command and waiting.
- Bulk mechanical edits use `python3` heredoc scripts. Why: `perl` is absent on this NixOS host, and `sed` cannot re-indent multi-line blocks. How to apply: reshaping call sites or nested literals across a file.

## Gotchas

- Bare model ids like `claude-sonnet-4-5` resolve ambiguously once `ModelRuntime` loads Pi's catalogs, because several providers expose the same id. Qualified `provider/id` is the reliable form.
- `ModelRuntime.create({ refreshOnCreate: false })` leaves `hasConfiguredAuth()` false for a provider present in `auth.json`, because the availability snapshot never runs. `listCredentials()` still reflects the file.

## Decisions

- Chat inference is stateless and calls pi-ai directly because every request carries its complete conversation; `AgentSession`, RPC subprocesses, and application session caching add machinery without needed state.
- The router takes `ModelRuntime` from `@earendil-works/pi-coding-agent` rather than `pi-ai` alone, because its `AuthStorage` holds a `proper-lockfile` lock on `~/.pi/agent/auth.json`, so Pi and the router cannot lose each other's OAuth refreshes.
- The 11 MB `pi-coding-agent` dependency was accepted over adding a ~40-line lock to a router-local credential store, to avoid mirroring Pi's lock convention by hand and drifting from it.
- Thinking signatures travel in OpenRouter's `reasoning_details` shape (`reasoning.text` with `signature`, `reasoning.encrypted` with `data`), because OpenAI defines no field for them and that shape is the closest de-facto standard.
- Inbound `reasoning_content` is ignored while `reasoning_details` is replayed, because text without its signature cannot re-enter a thinking block; pi-ai demotes it to plain text anyway.
- Reasoning round-trip only helps clients that echo `reasoning_details` back; many drop unknown response fields, and for those the router behaves as before.
- `createChatServer` takes an injected `Models` collection instead of building a default one, so the factory stays synchronous despite `ModelRuntime.create()` being async, and tests inject fakes.

## Dead Ends

- ✗ Router-local `FileCredentialStore` writing `$XDG_CONFIG_HOME/pi-router/auth.json`, abandoned: it serialized refreshes only within one process, forcing manual credential copying to avoid racing Pi.
- ✗ Waiting for `pi-ai` to ship a file-backed credential store, abandoned: as of 0.84.2 it exports only `InMemoryCredentialStore` and expects the app to inject persistence.
