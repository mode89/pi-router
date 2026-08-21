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
    const toolNames = indexToolCallNames(messages);
    const conversation = [];
    for (const message of messages) {
        if (message?.role === "user") {
            conversation.push({
                role: "user",
                content: extractText(message.content),
            });
        } else if (message?.role === "assistant") {
            conversation.push(toAssistantEntry(message));
        } else if (message?.role === "tool") {
            conversation.push(toToolResultEntry(message, toolNames));
        }
    }
    return { systemPrompt, conversation };
}

// A tool message names its tool by id only, so collect the names up front;
// a result may then precede its call without changing the outcome.
function indexToolCallNames(messages) {
    const names = new Map();
    for (const message of messages) {
        if (!Array.isArray(message?.tool_calls)) continue;
        for (const call of message.tool_calls) {
            if (
                typeof call?.id === "string"
                && typeof call.function?.name === "string"
            ) {
                names.set(call.id, call.function.name);
            }
        }
    }
    return names;
}

function toAssistantEntry(message) {
    const toolCalls = parseToolCalls(message.tool_calls);
    const text = extractText(message.content);
    const textParts = text.length === 0 && toolCalls.length > 0
        ? []
        : [{ type: "text", text }];
    return {
        role: "assistant",
        content: [...textParts, ...toolCalls],
        stopReason: toolCalls.length > 0 ? "toolUse" : "stop",
    };
}

function toToolResultEntry(message, toolNames) {
    const toolCallId = message.tool_call_id;
    if (typeof toolCallId !== "string" || toolCallId.length === 0) {
        throw new RequestError("tool messages need tool_call_id");
    }
    const hasExplicitName = typeof message.name === "string"
        && message.name.length > 0;
    const toolName = hasExplicitName
        ? message.name
        : toolNames.get(toolCallId);
    if (toolName === undefined) {
        throw new RequestError(`unknown tool_call_id: ${toolCallId}`);
    }
    return {
        role: "toolResult",
        toolCallId,
        toolName,
        content: [{ type: "text", text: extractText(message.content) }],
        isError: false,
    };
}

function parseToolCalls(toolCalls) {
    if (!Array.isArray(toolCalls)) return [];
    return toolCalls.map((call) => {
        const id = call?.id;
        const name = call?.function?.name;
        if (typeof id !== "string" || typeof name !== "string") {
            throw new RequestError(
                "each tool_calls entry needs id and function.name",
            );
        }
        return {
            type: "toolCall",
            id,
            name,
            arguments: parseToolArguments(call.function.arguments, name),
        };
    });
}

function parseToolArguments(raw, name) {
    if (raw === undefined || raw === null || raw === "") return {};
    let parsed;
    try {
        parsed = JSON.parse(raw);
    } catch (error) {
        throw new RequestError(
            `invalid arguments for tool ${name}: ${error.message}`,
        );
    }
    if (
        parsed === null || typeof parsed !== "object" || Array.isArray(parsed)
    ) {
        throw new RequestError(`arguments for tool ${name} must be an object`);
    }
    return parsed;
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
        throw new RequestError("no user/assistant/tool messages");
    }
    const lastRole = conversation.at(-1).role;
    if (lastRole !== "user" && lastRole !== "toolResult") {
        throw new RequestError("last message must have role=user or role=tool");
    }

    return {
        requestedModel: body.model,
        model,
        systemPrompt,
        conversation,
        tools: parseTools(body.tools, body.tool_choice, logger),
        reasoning: normalizeReasoningEffort(body.reasoning_effort, logger),
        stream: body.stream === true,
        includeUsage: body.stream === true
            && body.stream_options?.include_usage === true,
    };
}

export function parseTools(tools, toolChoice, logger = console) {
    if (tools === undefined || tools === null) return undefined;
    if (!Array.isArray(tools)) throw new RequestError("tools must be an array");
    const parsed = tools.map(toToolDefinition);
    if (parsed.length === 0) return undefined;
    // pi-ai's completeSimple copies a fixed option list and drops toolChoice,
    // so "none" (send no tools) is the only choice the router can honor.
    if (toolChoice === "none") {
        logger.warn?.("honoring tool_choice=none: sending no tools");
        return undefined;
    }
    if (toolChoice !== undefined && toolChoice !== null
        && toolChoice !== "auto") {
        const formatted = JSON.stringify(toolChoice);
        logger.warn?.(`ignoring unsupported tool_choice: ${formatted}`);
    }
    return parsed;
}

function toToolDefinition(tool) {
    if (tool === null || typeof tool !== "object") {
        throw new RequestError("each tool must be an object");
    }
    if (tool.type !== undefined && tool.type !== "function") {
        throw new RequestError(`unsupported tool type: ${tool.type}`);
    }
    const declaration = tool.function;
    if (
        declaration === null || typeof declaration !== "object"
        || typeof declaration.name !== "string" || declaration.name.length === 0
    ) {
        throw new RequestError("each tool needs a non-empty function.name");
    }
    return {
        name: declaration.name,
        description: typeof declaration.description === "string"
            ? declaration.description
            : "",
        parameters: declaration.parameters
            ?? { type: "object", properties: {} },
    };
}

export function buildContext(chatRequest, timestamp = Date.now()) {
    const { systemPrompt, tools, model } = chatRequest;
    const messages = chatRequest.conversation.map((message) => (
        message.role === "assistant"
            ? { ...message, ...replayedAssistantFields(model), timestamp }
            : { ...message, timestamp }
    ));
    const context = { messages };
    if (systemPrompt) context.systemPrompt = systemPrompt;
    if (tools) context.tools = tools;
    return context;
}

// Replayed assistant turns carry no provenance or usage, but pi-ai's
// AssistantMessage requires both, so stand in for the original generation.
function replayedAssistantFields(model) {
    return {
        api: model.api,
        provider: model.provider,
        model: model.id,
        usage: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 0,
            cost: {
                input: 0,
                output: 0,
                cacheRead: 0,
                cacheWrite: 0,
                total: 0,
            },
        },
    };
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
    const toolCalls = toolCallParts(assistant.content).map(toFunctionCall);
    const hasToolCalls = toolCalls.length > 0;
    const message = {
        role: "assistant",
        content: hasToolCalls && text.length === 0 ? null : text,
    };
    if (reasoning.length > 0) message.reasoning_content = reasoning;
    if (hasToolCalls) message.tool_calls = toolCalls;
    return {
        id: `chatcmpl-${id}`,
        object: "chat.completion",
        created,
        model: requestedModel,
        choices: [{
            index: 0,
            message,
            finish_reason: finishReason(assistant.stopReason, hasToolCalls),
        }],
        usage: toUsage(assistant.usage),
    };
}

function toolCallParts(content) {
    return (content ?? []).filter((part) => part?.type === "toolCall");
}

function toFunctionCall(toolCall) {
    return {
        id: toolCall.id,
        type: "function",
        function: {
            name: toolCall.name,
            arguments: JSON.stringify(toolCall.arguments ?? {}),
        },
    };
}

function toUsage(usage) {
    const promptTokens = Math.trunc(usage?.input ?? 0);
    const completionTokens = Math.trunc(usage?.output ?? 0);
    return {
        prompt_tokens: promptTokens,
        completion_tokens: completionTokens,
        total_tokens: promptTokens + completionTokens,
    };
}

// The fields every chunk of one streamed response repeats verbatim.
export function chunkEnvelope(
    requestedModel,
    {
        id = randomUUID().replaceAll("-", ""),
        created = Math.floor(Date.now() / 1000),
    } = {},
) {
    return {
        id: `chatcmpl-${id}`,
        object: "chat.completion.chunk",
        created,
        model: requestedModel,
    };
}

// One event maps to zero, one, or (with usage requested) two chunks.
export function buildChunks(envelope, event, includeUsage) {
    if (event.type === "text_delta") {
        return [toChunk(envelope, { content: event.delta })];
    }
    if (event.type === "thinking_delta") {
        return [toChunk(envelope, { reasoning_content: event.delta })];
    }
    // Tool calls go out whole at toolcall_end rather than as argument
    // fragments: some providers have no call id yet at toolcall_start, so
    // earlier fragments cannot be addressed to a call.
    if (event.type === "toolcall_end") {
        return [toChunk(envelope, { tool_calls: [streamedToolCall(event)] })];
    }
    if (event.type !== "done" && event.type !== "error") return [];
    const message = event.type === "done" ? event.message : event.error;
    const chunks = [finalChunk(envelope, message)];
    if (includeUsage) chunks.push(usageChunk(envelope, message));
    return chunks;
}

function streamedToolCall(event) {
    return {
        index: toolCallIndex(event),
        ...toFunctionCall(event.toolCall),
    };
}

// OpenAI numbers tool calls among themselves, not among all content parts.
function toolCallIndex(event) {
    const preceding = event.partial.content.slice(0, event.contentIndex);
    return toolCallParts(preceding).length;
}

function finalChunk(envelope, message) {
    const hasToolCalls = toolCallParts(message.content).length > 0;
    return toChunk(
        envelope,
        {},
        finishReason(message.stopReason, hasToolCalls),
    );
}

// OpenAI reports streamed usage in a trailing chunk that carries no choice.
function usageChunk(envelope, message) {
    return {
        ...envelope,
        choices: [],
        usage: toUsage(message.usage),
    };
}

function toChunk(envelope, delta, reason = null) {
    return {
        ...envelope,
        choices: [{ index: 0, delta, finish_reason: reason }],
    };
}

function finishReason(stopReason, hasToolCalls) {
    // hasToolCalls also decides it, for providers that emit tool calls
    // without setting stopReason=toolUse.
    if (stopReason === "toolUse" || hasToolCalls) return "tool_calls";
    return stopReason === "length" ? "length" : "stop";
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
            if (chatRequest.stream) {
                await streamCompletion(models, chatRequest, response, logger);
                return;
            }
            const assistant = await models.completeSimple(
                chatRequest.model,
                buildContext(chatRequest),
                completionOptions(chatRequest),
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

async function streamCompletion(models, chatRequest, response, logger) {
    const controller = new globalThis.AbortController();
    // close also fires on a normal end; aborting a finished stream is a no-op.
    response.on("close", () => controller.abort());
    const events = models.streamSimple(
        chatRequest.model,
        buildContext(chatRequest),
        { ...completionOptions(chatRequest), signal: controller.signal },
    );

    const envelope = chunkEnvelope(chatRequest.requestedModel);
    beginEventStream(response);
    // The opening role chunk is ours, not a mapped `start` event: adapters may
    // skip `start`, and OpenAI clients expect the role before any delta.
    sendChunk(response, toChunk(envelope, { role: "assistant", content: "" }));
    try {
        for await (const event of events) {
            if (event.type === "error") {
                // pi-ai reports a failed turn as its final AssistantMessage.
                warnOnFailedStop(event.error, logger);
            }
            const chunks = buildChunks(
                envelope,
                event,
                chatRequest.includeUsage,
            );
            for (const chunk of chunks) sendChunk(response, chunk);
        }
    } catch (error) {
        // The status line is already sent, so no error status can follow.
        // OpenAI defines no failing finish_reason, so the client sees "stop"
        // and a short answer; the real cause stays in the server log.
        logger.error?.("stream failed", error);
        sendChunk(response, toChunk(envelope, {}, "stop"));
    }
    endEventStream(response);
}

function completionOptions(chatRequest) {
    return chatRequest.reasoning === undefined
        ? {}
        : { reasoning: chatRequest.reasoning };
}

function beginEventStream(response) {
    response.writeHead(200, {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    });
}

function sendChunk(response, chunk) {
    response.write(`data: ${JSON.stringify(chunk)}\n\n`);
}

function endEventStream(response) {
    response.end("data: [DONE]\n\n");
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
