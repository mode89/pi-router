import http from "node:http";
import { Buffer } from "node:buffer";
import console from "node:console";
import { randomUUID } from "node:crypto";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";

const IGNORED_REQUEST_FIELDS = [
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

export class UsageError extends Error {}

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

export function normalizeReasoningEffort(effort, logger = console) {
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
        const model = models.getModel(provider, modelId);
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
    for (const field of IGNORED_REQUEST_FIELDS) {
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
        reasoning: normalizeReasoningEffort(body.reasoning_effort, logger),
    };
}

export function buildContext(chatRequest, timestamp = Date.now()) {
    const zeroUsage = () => ({
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
    });
    const messages = chatRequest.conversation.map((message) => {
        if (message.role === "user") {
            return { role: "user", content: message.content, timestamp };
        }
        return {
            role: "assistant",
            content: [{ type: "text", text: message.content }],
            api: chatRequest.model.api,
            provider: chatRequest.model.provider,
            model: chatRequest.model.id,
            usage: zeroUsage(),
            stopReason: "stop",
            timestamp,
        };
    });
    return chatRequest.systemPrompt
        ? { systemPrompt: chatRequest.systemPrompt, messages }
        : { messages };
}

export function buildCompletion(
    requestedModel,
    assistant,
    {
        id = randomUUID().replaceAll("-", ""),
        created = Math.floor(Date.now() / 1000),
    } = {},
) {
    const text = joinParts(assistant.content, "text", "text");
    const reasoning = joinParts(
        assistant.content,
        "thinking",
        "thinking",
        "\n\n",
    );
    const promptTokens = Math.trunc(assistant.usage?.input ?? 0);
    const completionTokens = Math.trunc(assistant.usage?.output ?? 0);
    const message = { role: "assistant", content: text };
    if (reasoning.length > 0) message.reasoning_content = reasoning;
    return {
        id: `chatcmpl-${id}`,
        object: "chat.completion",
        created,
        model: requestedModel,
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

function joinParts(content, type, field, separator = "") {
    return content
        .filter((part) => (
            part?.type === type && typeof part[field] === "string"
        ))
        .map((part) => part[field])
        .join(separator);
}

export function createChatServer({ models, logger = console }) {
    return http.createServer(async (request, response) => {
        if (request.method !== "POST" || request.url !== "/chat/completions") {
            sendError(response, 404, "not found", "invalid_request_error");
            return;
        }

        try {
            const body = parseJson(await readBody(request));
            const chatRequest = parseRequest(body, models, logger);
            const completionOptions = chatRequest.reasoning === undefined
                ? {}
                : { reasoning: chatRequest.reasoning };
            const assistant = await models.completeSimple(
                chatRequest.model,
                buildContext(chatRequest),
                completionOptions,
            );
            warnOnFailedStop(assistant, logger);
            sendJson(response, 200, buildCompletion(
                chatRequest.requestedModel,
                assistant,
            ));
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

function parseJson(raw) {
    try {
        return JSON.parse(raw);
    } catch (error) {
        throw new RequestError(`invalid JSON: ${error.message}`);
    }
}

function warnOnFailedStop(assistant, logger) {
    if (
        assistant.stopReason !== "error"
        && assistant.stopReason !== "aborted"
    ) {
        return;
    }
    const detail = assistant.errorMessage || "(no detail)";
    logger.warn?.(`assistant stopReason=${assistant.stopReason}: ${detail}`);
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

export function parseArgs(args) {
    let host = "127.0.0.1";
    let port = "8742";
    let help = false;

    for (let index = 0; index < args.length; index += 1) {
        const argument = args[index];
        const equals = argument.indexOf("=");
        const name = equals === -1 ? argument : argument.slice(0, equals);
        const attached = equals === -1
            ? undefined
            : argument.slice(equals + 1);
        const takeValue = () => {
            if (attached !== undefined) return attached;
            index += 1;
            if (index >= args.length) {
                throw new UsageError(`missing value for ${name}`);
            }
            return args[index];
        };

        if (name === "--help" || name === "-h") {
            help = true;
        } else if (name === "--host") {
            host = takeValue();
        } else if (name === "--port") {
            port = takeValue();
        } else {
            throw new UsageError(`unknown argument: ${argument}`);
        }
    }

    const portNumber = Number(port);
    if (
        !Number.isInteger(portNumber) || portNumber < 0 || portNumber > 65535
    ) {
        throw new UsageError(`invalid port: ${port}`);
    }
    if (!host) throw new UsageError("host must not be empty");
    return { host, port: portNumber, help };
}

export async function main(args = process.argv.slice(2)) {
    const options = parseArgs(args);
    if (options.help) {
        process.stdout.write("Usage: pirouter [--host HOST] [--port PORT]\n");
        return;
    }

    // Cached catalogs suffice; a catalog fetch must not delay listening.
    const models = await ModelRuntime.create({ allowModelNetwork: false });
    const server = createChatServer({ models });
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(options.port, options.host, resolve);
    });
    const { port } = server.address();
    console.info(`pirouter listening on http://${options.host}:${port}`);

    await new Promise((resolve) => {
        const stop = () => server.close(resolve);
        process.once("SIGINT", stop);
        process.once("SIGTERM", stop);
    });
}

const isDirectExecution = process.argv[1]
    && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isDirectExecution) {
    main().catch((error) => {
        console.error(`pirouter: ${error.message}`);
        process.exitCode = 1;
    });
}
