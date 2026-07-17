import http from "node:http";
import { Buffer } from "node:buffer";
import console from "node:console";
import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, unlink, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { dirname, join } from "node:path";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { builtinModels } from "@earendil-works/pi-ai/providers/all";

export const IGNORED_WITH_WARNING = [
    "temperature",
    "stop",
    "max_tokens",
    "max_completion_tokens",
];

const POSITIVE_REASONING_EFFORTS = new Set([
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]);

export class RequestError extends Error {}

export class FileCredentialStore {
    constructor(path = credentialPath()) {
        this.path = path;
        this.mutation = Promise.resolve();
    }

    async read(providerId) {
        return (await readCredentials(this.path))[providerId];
    }

    modify(providerId, fn) {
        const result = this.mutation.catch(() => {}).then(async () => {
            const credentials = await readCredentials(this.path);
            const current = credentials[providerId];
            const replacement = await fn(current);
            if (replacement === undefined) return current;
            await writeCredentials(this.path, {
                ...credentials,
                [providerId]: replacement,
            });
            return replacement;
        });
        this.mutation = result;
        return result;
    }
}

async function readCredentials(path) {
    try {
        return JSON.parse(await readFile(path, "utf8"));
    } catch (error) {
        if (error.code === "ENOENT") return {};
        throw error;
    }
}

async function writeCredentials(path, credentials) {
    const directory = dirname(path);
    const temporary = join(directory, `.${randomUUID()}.tmp`);
    await mkdir(directory, { recursive: true, mode: 0o700 });
    try {
        await writeFile(
            temporary,
            `${JSON.stringify(credentials, null, 2)}\n`,
            { encoding: "utf8", mode: 0o600 },
        );
        await rename(temporary, path);
    } catch (error) {
        await unlink(temporary).catch(() => {});
        throw error;
    }
}

export function credentialPath(env = process.env, home = homedir()) {
    const configHome = env.XDG_CONFIG_HOME || join(home, ".config");
    return join(configHome, "pi-router", "auth.json");
}

export function extractText(content) {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
        .filter((part) => (
            part !== null && typeof part === "object" && part.type === "text"
        ))
        .map((part) => part.text)
        .filter((text) => typeof text === "string")
        .join("");
}

export function splitMessages(messages) {
    const systemPrompt = messages
        .filter((message) => (
            message?.role === "system" || message?.role === "developer"
        ))
        .map((message) => extractText(message.content))
        .filter((text) => text.length > 0)
        .join("\n\n");
    const conversation = messages
        .filter((message) => (
            message?.role === "user" || message?.role === "assistant"
        ))
        .map((message) => ({
            role: message.role,
            content: extractText(message.content),
        }));
    return { systemPrompt, conversation };
}

export function lookupReasoning(effort, logger = console) {
    if (effort === undefined || effort === null || effort === "none") {
        return undefined;
    }
    if (POSITIVE_REASONING_EFFORTS.has(effort)) return effort;
    const formattedEffort = JSON.stringify(effort);
    logger.warn?.(`unknown reasoning_effort: ${formattedEffort} (ignored)`);
    return undefined;
}

export function resolveModel(models, requestedModel) {
    if (typeof requestedModel !== "string" || requestedModel.length === 0) {
        throw new RequestError("model must be a non-empty string");
    }

    const slash = requestedModel.indexOf("/");
    if (slash !== -1) {
        const provider = requestedModel.slice(0, slash);
        const modelId = requestedModel.slice(slash + 1);
        if (!provider || !modelId) {
            throw new RequestError(`unknown model: ${requestedModel}`);
        }
        const model = models.getModel?.(provider, modelId)
            ?? models.getModels(provider).find((candidate) => (
                candidate.provider === provider && candidate.id === modelId
            ));
        if (!model) throw new RequestError(`unknown model: ${requestedModel}`);
        return model;
    }

    const matches = models
        .getModels()
        .filter((model) => model.id === requestedModel);
    if (matches.length === 0) {
        throw new RequestError(`unknown model: ${requestedModel}`);
    }
    if (matches.length > 1) {
        throw new RequestError(`ambiguous model: ${requestedModel}`);
    }
    return matches[0];
}

export function parseRequest(body, models, logger = console) {
    if (body === null || typeof body !== "object" || Array.isArray(body)) {
        throw new RequestError("request body must be a JSON object");
    }
    if (!Array.isArray(body.messages) || body.messages.length === 0) {
        throw new RequestError("messages must be a non-empty array");
    }
    for (const field of IGNORED_WITH_WARNING) {
        if (Object.hasOwn(body, field)) {
            logger.warn?.(`ignoring unsupported field: ${field}`);
        }
    }

    const model = resolveModel(models, body.model);
    const { systemPrompt, conversation } = splitMessages(body.messages);
    if (conversation.length === 0) {
        throw new RequestError("no user/assistant messages");
    }
    if (conversation.at(-1).role !== "user") {
        throw new RequestError("last message must have role=user");
    }

    return {
        requestedModel: body.model,
        model,
        systemPrompt,
        conversation,
        reasoning: lookupReasoning(body.reasoning_effort, logger),
    };
}

export function buildContext(parsed, timestamp = Date.now()) {
    const zeroUsage = () => ({
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    });
    const messages = parsed.conversation.map((message) => {
        if (message.role === "user") {
            return { role: "user", content: message.content, timestamp };
        }
        return {
            role: "assistant",
            content: [{ type: "text", text: message.content }],
            api: parsed.model.api,
            provider: parsed.model.provider,
            model: parsed.model.id,
            usage: zeroUsage(),
            stopReason: "stop",
            timestamp,
        };
    });
    return parsed.systemPrompt
        ? { systemPrompt: parsed.systemPrompt, messages }
        : { messages };
}

export function buildCompletion(
    requestedModel,
    assistant,
    {
        id = randomUUID().replaceAll("-", ""),
        created = Math.floor(Date.now() / 1000),
        logger = console,
    } = {},
) {
    const text = assistant.content
        .filter((part) => (
            part?.type === "text" && typeof part.text === "string"
        ))
        .map((part) => part.text)
        .join("");
    const reasoning = assistant.content
        .filter((part) => (
            part?.type === "thinking" && typeof part.thinking === "string"
        ))
        .map((part) => part.thinking)
        .join("\n\n");
    if (
        assistant.stopReason === "error"
        || assistant.stopReason === "aborted"
    ) {
        const detail = assistant.errorMessage || "(no detail)";
        const stopReason = assistant.stopReason;
        logger.warn?.(`assistant stopReason=${stopReason}: ${detail}`);
    }

    const promptTokens = Math.trunc(assistant.usage?.input ?? 0);
    const completionTokens = Math.trunc(assistant.usage?.output ?? 0);
    const message = { role: "assistant", content: text };
    if (reasoning.length > 0) message.reasoning_content = reasoning;
    return {
        id: `chatcmpl-${id}`,
        object: "chat.completion",
        created,
        model: requestedModel || "",
        choices: [{
            index: 0,
            message,
            finish_reason: assistant.stopReason === "length"
                ? "length"
                : "stop",
        }],
        usage: {
            prompt_tokens: promptTokens,
            completion_tokens: completionTokens,
            total_tokens: promptTokens + completionTokens,
        },
    };
}

export function createChatServer({
    models,
    credentials,
    logger = console,
} = {}) {
    const modelCollection = models ?? builtinModels({
        credentials: credentials ?? new FileCredentialStore(),
    });
    return http.createServer(async (request, response) => {
        if (request.method !== "POST" || request.url !== "/chat/completions") {
            sendError(response, 404, "not found", "invalid_request_error");
            return;
        }

        try {
            const raw = await readBody(request);
            let body;
            try {
                body = JSON.parse(raw);
            } catch (error) {
                sendError(
                    response,
                    400,
                    `invalid JSON: ${error.message}`,
                    "invalid_request_error",
                );
                return;
            }
            const parsed = parseRequest(body, modelCollection, logger);
            const context = buildContext(parsed);
            const options = parsed.reasoning === undefined
                ? {}
                : { reasoning: parsed.reasoning };
            const assistant = await modelCollection.completeSimple(
                parsed.model,
                context,
                options,
            );
            const completion = buildCompletion(
                parsed.requestedModel,
                assistant,
                { logger },
            );
            sendJson(response, 200, completion);
        } catch (error) {
            if (error instanceof RequestError) {
                sendError(
                    response,
                    400,
                    error.message,
                    "invalid_request_error",
                );
            } else {
                logger.error?.("request failed", error);
                const message = error instanceof Error
                    ? error.message
                    : String(error);
                sendError(response, 500, message, "server_error");
            }
        }
    });
}

export function parseArgs(args) {
    const options = { host: "127.0.0.1", port: 8742, help: false };
    for (let index = 0; index < args.length; index += 1) {
        const argument = args[index];
        if (argument === "--help" || argument === "-h") {
            options.help = true;
        } else if (argument === "--host" || argument === "--port") {
            if (index + 1 >= args.length) {
                throw new RequestError(`missing value for ${argument}`);
            }
            options[argument.slice(2)] = args[++index];
        } else if (argument.startsWith("--host=")) {
            options.host = argument.slice(7);
        } else if (argument.startsWith("--port=")) {
            options.port = argument.slice(7);
        } else {
            throw new RequestError(`unknown argument: ${argument}`);
        }
    }
    const port = Number(options.port);
    if (!Number.isInteger(port) || port < 0 || port > 65535) {
        throw new RequestError(`invalid port: ${options.port}`);
    }
    if (!options.host) throw new RequestError("host must not be empty");
    return { ...options, port };
}

export async function main(args = process.argv.slice(2)) {
    const options = parseArgs(args);
    if (options.help) {
        process.stdout.write("Usage: pirouter [--host HOST] [--port PORT]\n");
        return;
    }

    const server = createChatServer();
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(options.port, options.host, resolve);
    });
    const address = server.address();
    const port = typeof address === "object" ? address.port : options.port;
    console.info(`pirouter listening on http://${options.host}:${port}`);

    await new Promise((resolve) => {
        let stopping = false;
        const stop = () => {
            if (stopping) return;
            stopping = true;
            server.close(resolve);
        };
        process.once("SIGINT", stop);
        process.once("SIGTERM", stop);
    });
}

function readBody(request) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        request.setEncoding("utf8");
        request.on("data", (chunk) => chunks.push(chunk));
        request.on("end", () => resolve(chunks.join("")));
        request.on("error", reject);
        request.on("aborted", () => reject(new Error("request aborted")));
    });
}

function sendJson(response, status, payload) {
    const body = JSON.stringify(payload);
    response.writeHead(status, {
        "Content-Type": "application/json",
        "Content-Length": Buffer.byteLength(body),
    });
    response.end(body);
}

function sendError(response, status, message, type) {
    sendJson(response, status, { error: { message, type, code: status } });
}

const isDirectExecution = process.argv[1]
  && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isDirectExecution) {
    main().catch((error) => {
        console.error(`pirouter: ${error.message}`);
        process.exitCode = 1;
    });
}
