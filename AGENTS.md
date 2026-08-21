# pi-router

Stateless Node.js router exposing an OpenAI-compatible `POST /chat/completions` endpoint, buffered or streamed as SSE, backed by `@earendil-works/pi-coding-agent`'s `ModelRuntime`.

## Requirements

- Node.js 22.19 or newer
- npm with dependencies installed from `package-lock.json`

## Key files

- `pirouter.js`: HTTP server, request translation, and model inference
- `pirouter.test.js`: unit, HTTP, credential, and launcher tests
- `pi-router`: executable launcher that installs missing runtime dependencies
- `eslint.config.js`: lint configuration

## Commands

- `npm install`: install all dependencies
- `npm test`: run the Node test suite
- `npm run lint`: run ESLint
- `npm start`: listen on `127.0.0.1:8742`
- `./pi-router [--host HOST] [--port PORT]`: install missing runtime dependencies and launch

## Runtime configuration

`ModelRuntime` reads credentials from `~/.pi/agent/auth.json` and custom models from `~/.pi/agent/models.json`, the same files Pi itself uses.

**Memory — read first.** Read `MEMORY.md` at the start of each session, before your first response — it records facts about this project, its conventions, landmines, dead ends, and decision rationale you can't recover from the code. Skipping it risks repeating solved mistakes.
