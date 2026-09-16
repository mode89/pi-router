import http from "node:http";
import { Buffer } from "node:buffer";
import console from "node:console";
import { randomUUID } from "node:crypto";
import process from "node:process";
import { pathToFileURL } from "node:url";
import { ModelRuntime, resolveCliModel } from "@earendil-works/pi-coding-agent";

export const CHAT_COMPLETIONS_PATH = "/api/v1/chat/completions";

const RECOGNIZED_REQUEST_FIELDS = new Set([
    "model",
    "messages",
    "tools",
    "tool_choice",
    "reasoning",
    "response_format",
    "stream",
    "stream_options",
]);

const POSITIVE_REASONING_EFFORTS = new Set([
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
]);

export class RequestError extends Error {}

class ProviderError extends Error {
    constructor(message, status) {
        super(message);
        this.status = status;
    }
}

export class UsageError extends Error {}

export function extractText(content) {
    if (typeof content === "string") return content;
    if (!Array.isArray(content)) return "";
    return content
        .map(textOfPart)
        .filter((text) => typeof text === "string")
        .join("");
}

function textOfPart(part) {
    if (part === null || typeof part !== "object") return "";
    rejectNonTextPart(part);
    return part.text;
}

// Only text reaches the model, so any other part, an image above all, is
// refused: answering as if the model had seen it would mislead the client.
function rejectNonTextPart(part) {
    if (part.type !== "text") {
        throw new RequestError(
            `unsupported message content part: ${part.type}`,
        );
    }
}

export function splitMessages(messages, model) {
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
            conversation.push(toAssistantEntry(message, model));
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

function toAssistantEntry(message, model) {
    const toolCalls = parseToolCalls(message.tool_calls);
    const text = extractText(message.content);
    const textParts = text.length === 0 && toolCalls.length > 0
        ? []
        : [{ type: "text", text }];
    const reasoning = replayedReasoning(message.reasoning_details, model);
    return {
        role: "assistant",
        // Anthropic rejects a turn whose thinking block does not come first.
        content: [
            ...reasoning.parts,
            ...textParts,
            ...toolCalls,
        ],
        stopReason: toolCalls.length > 0 ? "toolUse" : "stop",
        ...replayedAssistantFields(reasoning.origin ?? UNVERIFIED_MODEL),
    };
}

// No real model carries this identity, so pi-ai treats such a turn's reasoning
// as foreign and downgrades it. A turn whose producing model pi-router cannot
// verify must not pass as the selected model's own work.
const UNVERIFIED_MODEL = {
    api: "pirouter-unverified",
    provider: "pirouter-unverified",
    id: "pirouter-unverified",
};

// A replayed assistant turn carries no usage, but pi-ai's AssistantMessage
// requires one, so stand in for the original generation. Its model identity is
// the one that produced the turn, which is what lets pi-ai protect reasoning
// against a model switch.
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

// A thinking block replays only with the signature its provider issued and
// only to that same model, so pi-router's signature carries both, and the turn
// keeps the identity of the model that produced it. Plain reasoning_content is
// text alone and cannot be replayed.
function replayedReasoning(details, selectedModel) {
    if (!Array.isArray(details)) return { parts: [], origin: undefined };
    const provenances = details.map((detail) => (
        decodeThinkingSignature(detailSignature(detail))
    ));
    const origin = provenances
        .find((provenance) => provenance !== undefined)
        ?.model;
    const parts = details
        .map((detail, index) => toThinkingPart(
            detail,
            replayedSignature(provenances[index], origin, selectedModel),
        ))
        .filter((part) => part !== null);
    return { parts, origin };
}

function toThinkingPart(detail, signature) {
    if (detail?.type === "reasoning.encrypted") {
        // Redacted reasoning is nothing but its payload, so without a payload
        // this model can replay there is no block left to send.
        if (signature === undefined || signature.length === 0) return null;
        return {
            type: "thinking",
            // Placeholder text only: the adapter replays the encrypted payload
            // from thinkingSignature and discards this.
            thinking: "[Reasoning redacted]",
            thinkingSignature: signature,
            redacted: true,
        };
    }
    if (detail?.type !== "reasoning.text"
        || typeof detail.text !== "string") {
        return null;
    }
    const part = { type: "thinking", thinking: detail.text };
    if (signature !== undefined && signature.length > 0) {
        part.thinkingSignature = signature;
    }
    return part;
}

// On OpenAI-compatible routes pi-ai stores the name of the response field that
// carried reasoning where other routes store a signature, and replays it as a
// field name, so only pi-router's own values may return to it.
const SIGNATURE_PREFIX = "pirouter-v1.";

// What pi-router's signature carries: the identity of the model that produced
// a thinking block, and the value pi-ai stored for that block.
function thinkingProvenance(model, signature) {
    return { model, signature };
}

function encodeThinkingSignature(model, signature) {
    const payload = JSON.stringify({
        api: model.api,
        provider: model.provider,
        id: model.id,
        signature,
    });
    const encoded = Buffer.from(payload, "utf8").toString("base64url");
    return `${SIGNATURE_PREFIX}${encoded}`;
}

function decodeThinkingSignature(encoded) {
    if (typeof encoded !== "string" || !encoded.startsWith(SIGNATURE_PREFIX)) {
        return undefined;
    }
    const payload = parseSignaturePayload(
        encoded.slice(SIGNATURE_PREFIX.length),
    );
    if (payload === undefined) return undefined;
    const { api, provider, id, signature } = payload;
    if ([api, provider, id, signature].some((field) => (
        typeof field !== "string"
    ))) {
        return undefined;
    }
    return thinkingProvenance({ api, provider, id }, signature);
}

function parseSignaturePayload(encoded) {
    try {
        const payload = JSON.parse(
            Buffer.from(encoded, "base64url").toString("utf8"),
        );
        return payload !== null && typeof payload === "object"
            ? payload
            : undefined;
    } catch {
        return undefined;
    }
}

function detailSignature(detail) {
    if (detail?.type === "reasoning.encrypted") return detail.data;
    if (detail?.type === "reasoning.text") return detail.signature;
    return undefined;
}

// One replayed turn may mix entries from several models. Only the entries of
// the turn's origin may keep a signature, and only when that origin is the
// model now selected; every other entry travels as reasoning text alone.
function replayedSignature(provenance, origin, selectedModel) {
    if (provenance === undefined) return undefined;
    const belongsToTurn = sameModel(provenance.model, origin);
    return belongsToTurn && sameModel(provenance.model, selectedModel)
        ? provenance.signature
        : undefined;
}

function sameModel(produced, selected) {
    return produced.api === selected.api
        && produced.provider === selected.provider
        && produced.id === selected.id;
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

// OpenRouter's reasoning object; Pi's thinking levels are its effort values.
export function parseReasoningOption(reasoning, logger = console) {
    if (reasoning === undefined || reasoning === null) return undefined;
    if (typeof reasoning !== "object" || Array.isArray(reasoning)) {
        const formatted = JSON.stringify(reasoning);
        logger.warn?.(`ignoring unsupported reasoning: ${formatted}`);
        return undefined;
    }
    for (const member of Object.keys(reasoning)) {
        if (member !== "effort" && member !== "enabled") {
            logger.warn?.(`ignoring unsupported reasoning member: ${member}`);
        }
    }
    // An effort pi-router knows decides alone; `enabled` decides only when the
    // request names no such effort.
    const effort = knownEffort(reasoning.effort, logger);
    if (effort === "none") return undefined;
    if (effort !== undefined) return effort;
    if (reasoning.enabled === true) return "medium";
    return undefined;
}

function knownEffort(effort, logger) {
    if (effort === undefined || effort === null) return undefined;
    if (effort === "none" || POSITIVE_REASONING_EFFORTS.has(effort)) {
        return effort;
    }
    const formatted = JSON.stringify(effort);
    logger.warn?.(`ignoring unknown reasoning effort: ${formatted}`);
    return undefined;
}

// Pi's own resolver, so a model name reaches the router as it reaches Pi.
export function resolveModel(models, requestedModel, logger = console) {
    if (typeof requestedModel !== "string" || requestedModel.length === 0) {
        throw new RequestError("model must be a non-empty string");
    }
    // resolveCliModel also reports a thinking level parsed from a
    // "model:level" name; it is left unread, since the reasoning object is the
    // only reasoning control of this API.
    const { model, warning, error } = resolveCliModel({
        cliModel: requestedModel,
        modelRuntime: models,
    });
    if (warning) logger.warn?.(warning);
    if (!model) throw new RequestError(error ?? `unknown model: ${requestedModel}`);
    return model;
}

export function parseRequest(body, models, logger = console) {
    if (body === null || typeof body !== "object" || Array.isArray(body)) {
        throw new RequestError("request body must be a JSON object");
    }
    if (!Array.isArray(body.messages) || body.messages.length === 0) {
        throw new RequestError("messages must be a non-empty array");
    }
    rejectResponseFormat(body.response_format);
    for (const field of Object.keys(body)) {
        if (!RECOGNIZED_REQUEST_FIELDS.has(field)) {
            logger.warn?.(`ignoring unsupported field: ${field}`);
        }
    }

    const model = resolveModel(models, body.model, logger);
    const { systemPrompt, conversation } = splitMessages(body.messages, model);
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
        reasoning: parseReasoningOption(body.reasoning, logger),
        stream: body.stream === true,
    };
}

// Nothing here constrains what the model writes, so a demanded format could
// only be promised, not kept.
function rejectResponseFormat(responseFormat) {
    if (responseFormat === undefined || responseFormat === null) return;
    const type = responseFormat.type;
    if (type === undefined || type === "text") return;
    throw new RequestError(`unsupported response_format: ${type}`);
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
    const { systemPrompt, tools } = chatRequest;
    const messages = chatRequest.conversation.map((message) => (
        { ...message, timestamp }
    ));
    const context = { messages };
    if (systemPrompt) context.systemPrompt = systemPrompt;
    if (tools) context.tools = tools;
    return context;
}

export function buildCompletion(
    chatRequest,
    assistant,
    {
        id = randomUUID().replaceAll("-", ""),
        created = Math.floor(Date.now() / 1000),
    } = {},
) {
    const { requestedModel, model } = chatRequest;
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
    const reasoningDetails = toReasoningDetails(assistant.content, model);
    if (reasoning.length > 0) message.reasoning = reasoning;
    if (reasoningDetails.length > 0) {
        message.reasoning_details = reasoningDetails;
    }
    if (hasToolCalls) message.tool_calls = toolCalls;
    return {
        id: `chatcmpl-${id}`,
        object: "chat.completion",
        created,
        model: requestedModel,
        choices: [{
            index: 0,
            message,
            finish_reason: finishReason(assistant, hasToolCalls),
        }],
        usage: toUsage(assistant.usage),
    };
}

function toReasoningDetails(content, model) {
    return content
        .filter((part) => part?.type === "thinking")
        .map((part, index) => toReasoningDetail(part, index, model));
}

function toReasoningDetail(part, index, model) {
    const attribution = {
        format: reasoningFormat(model),
        id: null,
        index,
    };
    if (part.redacted === true) {
        return {
            type: "reasoning.encrypted",
            data: encodeThinkingSignature(model, part.thinkingSignature ?? ""),
            ...attribution,
        };
    }
    const detail = { type: "reasoning.text", text: part.thinking ?? "" };
    if (part.thinkingSignature) {
        detail.signature = encodeThinkingSignature(
            model,
            part.thinkingSignature,
        );
    }
    return { ...detail, ...attribution };
}

// OpenRouter names the provider family that produced the reasoning.
function reasoningFormat(model) {
    if (model.api === "anthropic-messages"
        || model.api === "bedrock-converse-stream") {
        return "anthropic-claude-v1";
    }
    if (model.api === "google-generative-ai" || model.api === "google-vertex") {
        return "google-gemini-v1";
    }
    // Every pi-ai api name ending in "responses" is a route of OpenAI's
    // Responses API, whatever provider serves it.
    return model.api?.endsWith("responses")
        ? "openai-responses-v1"
        : "unknown";
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
    const promptTokens = usage.input + usage.cacheRead + usage.cacheWrite;
    const promptCost = usage.cost.input
        + usage.cost.cacheRead
        + usage.cost.cacheWrite;
    const openRouterUsage = {
        prompt_tokens: promptTokens,
        completion_tokens: usage.output,
        total_tokens: usage.totalTokens,
        prompt_tokens_details: {
            cached_tokens: usage.cacheRead,
            cache_write_tokens: usage.cacheWrite,
        },
        cost: usage.cost.total,
        cost_details: {
            upstream_inference_prompt_cost: promptCost,
            upstream_inference_completions_cost: usage.cost.output,
            upstream_inference_cost: usage.cost.total,
        },
    };
    if (usage.reasoning !== undefined) {
        openRouterUsage.completion_tokens_details = {
            reasoning_tokens: usage.reasoning,
        };
    }
    return openRouterUsage;
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

// One event maps to zero, one, or two chunks.
export function buildChunks(envelope, event, model) {
    if (event.type === "text_delta") {
        return [toChunk(envelope, { content: event.delta })];
    }
    if (event.type === "thinking_delta") {
        return [toChunk(envelope, { reasoning: event.delta })];
    }
    // The signature only exists once the block closes, and a redacted block
    // emits no delta at all, so both travel in the thinking_end chunk.
    if (event.type === "thinking_end") {
        const part = event.partial?.content?.[event.contentIndex];
        if (part?.type !== "thinking") return [];
        return [toChunk(envelope, {
            reasoning_details: [
                toReasoningDetail(part, reasoningIndex(event), model),
            ],
        })];
    }
    // Tool calls go out whole at toolcall_end rather than as argument
    // fragments: some providers have no call id yet at toolcall_start, so
    // earlier fragments cannot be addressed to a call.
    if (event.type === "toolcall_end") {
        return [toChunk(envelope, { tool_calls: [streamedToolCall(event)] })];
    }
    if (event.type !== "done" && event.type !== "error") return [];
    // A failure throws before chunks are built, so an error event reaching
    // here is blocked content, reported as a finished turn.
    const message = event.type === "done" ? event.message : event.error;
    const chunks = [finalChunk(envelope, message)];
    if (message.usage !== undefined) {
        chunks.push(usageChunk(envelope, message));
    }
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

// OpenRouter numbers reasoning entries among themselves in the same way.
function reasoningIndex(event) {
    return event.partial.content
        .slice(0, event.contentIndex)
        .filter((part) => part?.type === "thinking")
        .length;
}

function finalChunk(envelope, message) {
    const hasToolCalls = toolCallParts(message.content).length > 0;
    return toChunk(
        envelope,
        {},
        finishReason(message, hasToolCalls),
    );
}

// OpenRouter reports streamed usage in a trailing chunk whose single choice
// carries an empty delta.
function usageChunk(envelope, message) {
    return {
        ...toChunk(envelope, {}),
        usage: toUsage(message.usage),
    };
}

function toChunk(envelope, delta, reason = null) {
    return {
        ...envelope,
        choices: [{ index: 0, delta, finish_reason: reason }],
    };
}

function finishReason(message, hasToolCalls) {
    if (isContentFiltered(message)) return "content_filter";
    // hasToolCalls also decides it, for providers that emit tool calls
    // without setting stopReason=toolUse.
    if (message.stopReason === "toolUse" || hasToolCalls) return "tool_calls";
    return message.stopReason === "length" ? "length" : "stop";
}

// Every provider reports blocked content as a failed turn, and keeps its own
// word for the block in rawStopReason.
const BLOCKED_STOP_REASONS = new Set([
    "content_filter",
    "sensitive",
    "refusal",
    "safety",
    "image_safety",
    "blocklist",
    "prohibited_content",
    "image_prohibited_content",
    "spii",
]);

function isContentFiltered(message) {
    return message.stopReason === "error"
        && BLOCKED_STOP_REASONS.has(
            String(message.rawStopReason ?? "").toLowerCase(),
        );
}

// A turn Pi reports as failed is a failure of the request, not an answer.
function providerFailure(message) {
    const failed = message.stopReason === "error"
        || message.stopReason === "aborted";
    if (!failed || isContentFiltered(message)) return undefined;
    const detail = message.errorMessage
        || `model stopped: ${message.stopReason}`;
    return new ProviderError(detail, providerStatus(detail));
}

function toProviderError(error) {
    if (error instanceof ProviderError) return error;
    const text = error instanceof Error ? error.message : String(error);
    return new ProviderError(text, providerStatus(text));
}

// Pi composes a provider error text that begins with the HTTP status it got.
function providerStatus(text) {
    const match = /^\s*(\d{3})\b/.exec(text);
    const status = match ? Number(match[1]) : 0;
    return status >= 400 && status <= 599 ? status : 500;
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
        if (request.method !== "POST"
            || request.url !== CHAT_COMPLETIONS_PATH) {
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
            const failure = providerFailure(assistant);
            if (failure) throw failure;
            sendJson(response, 200, buildCompletion(chatRequest, assistant));
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
                const failure = toProviderError(error);
                sendError(
                    response,
                    failure.status,
                    failure.message,
                    "provider_error",
                );
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
    const sink = chunkSink(response, envelope);
    try {
        for await (const event of events) {
            const failure = streamedFailure(event);
            if (failure) throw failure;
            const chunks = buildChunks(envelope, event, chatRequest.model);
            for (const chunk of chunks) sink.write(chunk);
        }
        sink.open();
    } catch (error) {
        logger.error?.("stream failed", error);
        const failure = toProviderError(error);
        if (!sink.isOpen()) {
            sendError(
                response,
                failure.status,
                failure.message,
                "provider_error",
            );
            return;
        }
        sink.write(errorChunk(failure));
    }
    endEventStream(response);
}

// The response head is written once, on the first chunk, so a failure before
// any output can still be answered with an HTTP status.
function chunkSink(response, envelope) {
    let opened = false;
    return {
        open() {
            if (opened) return;
            opened = true;
            beginEventStream(response);
            // The opening role chunk is ours, not a mapped `start` event:
            // adapters may skip `start`, and OpenAI clients expect the role
            // before any delta.
            sendChunk(
                response,
                toChunk(envelope, { role: "assistant", content: "" }),
            );
        },
        write(chunk) {
            this.open();
            sendChunk(response, chunk);
        },
        isOpen() {
            return opened;
        },
    };
}

// pi-ai reports a failed turn as an error event carrying the partial
// AssistantMessage; a done event always ends a turn that produced an answer.
function streamedFailure(event) {
    return event.type === "error" ? providerFailure(event.error) : undefined;
}

function errorChunk(failure) {
    return {
        error: {
            message: failure.message,
            type: "provider_error",
            code: failure.status,
        },
    };
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
