# pi-router

Stateless Node.js router exposing a non-streaming OpenAI-compatible `POST /chat/completions` endpoint backed by built-in `@earendil-works/pi-ai` models.

## Requirements

- Node.js 22.19 or newer
- npm with dependencies installed from `package-lock.json`

## Key files

- `pirouter.mjs`: HTTP server, request translation, model inference, and credential persistence
- `pirouter.test.mjs`: unit, HTTP, credential, and launcher tests
- `pi-router`: executable launcher that installs missing runtime dependencies
- `pirouter.pml`: original Paimel implementation retained as a reference
- `eslint.config.mjs`: lint configuration

## Commands

- `npm install`: install all dependencies
- `npm test`: run the Node test suite
- `npm run lint`: run ESLint
- `npm start`: listen on `127.0.0.1:8742`
- `./pi-router [--host HOST] [--port PORT]`: install missing runtime dependencies and launch

## Runtime configuration

Credentials are read from `$XDG_CONFIG_HOME/pi-router/auth.json`, falling back to `~/.config/pi-router/auth.json`. The file uses pi-ai's provider-keyed credential format.

**Memory — read first.** Read `MEMORY.md` at the start of each session, before your first response — it records facts about this project, its conventions, landmines, dead ends, and decision rationale you can't recover from the code. Skipping it risks repeating solved mistakes.
