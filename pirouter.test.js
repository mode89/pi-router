import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import {
    chmod,
    mkdtemp,
    rm,
    stat,
    writeFile,
} from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import process from "node:process";
import test from "node:test";
import { URL, fileURLToPath } from "node:url";
import { ModelRuntime } from "@earendil-works/pi-coding-agent";
import {
    UsageError,
    buildChunks,
    buildCompletion,
    buildContext,
    chunkEnvelope,
    createChatServer,
    extractText,
    normalizeReasoningEffort,
    parseArgs,
    parseRequest,
    parseTools,
    resolveModel,
    splitMessages,
} from "./pirouter.js";

const MODEL = {
    id: "model-a",
    provider: "provider-a",
    api: "test-api",
};

function fakeModels({
    models = [MODEL],
    complete = async () => assistantResult(),
    stream = async function* () { yield doneEvent(); },
} = {}) {
    return {
        calls: [],
        getModels(provider) {
            return provider === undefined
                ? models
                : models.filter((model) => model.provider === provider);
        },
        getModel(provider, id) {
            return models.find((model) => (
                model.provider === provider && model.id === id
            ));
        },
        async completeSimple(model, context, options) {
            this.calls.push({ model, context, options });
            return complete(model, context, options);
        },
        streamSimple(model, context, options) {
            this.calls.push({ model, context, options });
            return stream(model, context, options);
        },
    };
}

function doneEvent(overrides = {}) {
    return {
        type: "done",
        reason: "stop",
        message: assistantResult(overrides),
    };
}

function assistantResult(overrides = {}) {
    return {
        role: "assistant",
        content: [{ type: "text", text: "answer" }],
        api: MODEL.api,
        provider: MODEL.provider,
        model: MODEL.id,
        usage: {
            input: 7,
            output: 3,
            cacheRead: 0,
            cacheWrite: 0,
            totalTokens: 10,
            cost: {
                input: 0,
                output: 0,
                cacheRead: 0,
                cacheWrite: 0,
                total: 0,
            },
        },
        stopReason: "stop",
        timestamp: 1,
        ...overrides,
    };
}

test("content and message normalization preserve Paimel behavior", () => {
    assert.equal(extractText("plain"), "plain");
    assert.equal(extractText([
        { type: "text", text: "one" },
        { type: "image_url", image_url: "ignored" },
        { type: "text", text: "two" },
        { type: "text", text: 3 },
    ]), "onetwo");
    assert.equal(extractText(null), "");

    assert.deepEqual(splitMessages([
        { role: "system", content: "system" },
        { role: "developer", content: [{ type: "text", text: "developer" }] },
        { role: "assistant", content: "old answer" },
        { role: "user", content: [{ type: "text", text: "new question" }] },
    ]), {
        systemPrompt: "system\n\ndeveloper",
        conversation: [
            {
                role: "assistant",
                content: [{ type: "text", text: "old answer" }],
                stopReason: "stop",
            },
            { role: "user", content: "new question" },
        ],
    });
});

test("model resolution requires exact and unambiguous matches", () => {
    const duplicate = { ...MODEL, provider: "provider-b" };
    const models = fakeModels({
        models: [MODEL, duplicate, { ...MODEL, id: "other" }],
    });
    assert.equal(resolveModel(models, "provider-a/model-a"), MODEL);
    assert.equal(resolveModel(fakeModels(), "model-a"), MODEL);
    assert.throws(() => resolveModel(models, "model-a"), /ambiguous model/);
    assert.throws(
        () => resolveModel(models, "provider-a/missing"),
        /unknown model/,
    );
    assert.throws(() => resolveModel(models, "missing"), /unknown model/);
    assert.throws(
        () => resolveModel(models, ""),
        /model must be a non-empty string/,
    );
});

test("request validation, warnings, and reasoning mapping", () => {
    const warnings = [];
    const logger = { warn: (message) => warnings.push(message) };
    const parsed = parseRequest({
        model: "provider-a/model-a",
        temperature: 0.2,
        max_tokens: 10,
        reasoning_effort: "high",
        messages: [
            { role: "system", content: "rules" },
            { role: "user", content: "question" },
        ],
    }, fakeModels(), logger);
    assert.equal(parsed.reasoning, "high");
    assert.deepEqual(warnings, [
        "ignoring unsupported field: temperature",
        "ignoring unsupported field: max_tokens",
    ]);
    assert.equal(normalizeReasoningEffort("none"), undefined);
    assert.equal(normalizeReasoningEffort("minimal"), "minimal");
    assert.equal(normalizeReasoningEffort("max", logger), undefined);
    assert.match(warnings.at(-1), /unknown reasoning_effort/);

    const invalid = [
        [{}, /messages must be a non-empty array/],
        [
            { messages: [], model: "model-a" },
            /messages must be a non-empty array/,
        ],
        [
            {
                messages: [{ role: "system", content: "x" }],
                model: "model-a",
            },
            /no user\/assistant\/tool messages/,
        ],
        [
            {
                messages: [{ role: "assistant", content: "x" }],
                model: "model-a",
            },
            /last message must have role=user/,
        ],
        [
            {
                messages: [
                    { role: "tool", content: "x", tool_call_id: "gone" },
                ],
                model: "model-a",
            },
            /unknown tool_call_id: gone/,
        ],
        [
            { messages: [{ role: "user", content: "x" }] },
            /model must be a non-empty string/,
        ],
    ];
    for (const [body, expected] of invalid) {
        assert.throws(() => parseRequest(body, fakeModels(), logger), expected);
    }
});

test("native context synthesizes assistant history metadata", () => {
    const parsed = parseRequest({
        model: "provider-a/model-a",
        messages: [
            { role: "system", content: "rules" },
            { role: "user", content: "first" },
            { role: "assistant", content: "prior" },
            { role: "user", content: "next" },
        ],
    }, fakeModels());
    const context = buildContext(parsed, 1234);
    assert.equal(context.systemPrompt, "rules");
    assert.deepEqual(context.messages[0], {
        role: "user",
        content: "first",
        timestamp: 1234,
    });
    assert.deepEqual(context.messages[1], {
        role: "assistant",
        content: [{ type: "text", text: "prior" }],
        api: "test-api",
        provider: "provider-a",
        model: "model-a",
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
        stopReason: "stop",
        timestamp: 1234,
    });
    assert.deepEqual(context.messages[2], {
        role: "user",
        content: "next",
        timestamp: 1234,
    });
});

test("tool history becomes tool calls and tool results", () => {
    const parsed = parseRequest({
        model: "provider-a/model-a",
        messages: [
            { role: "user", content: "question" },
            {
                role: "assistant",
                content: null,
                tool_calls: [{
                    id: "call-1",
                    type: "function",
                    function: {
                        name: "lookup",
                        arguments: "{\"q\":\"pi\"}",
                    },
                }],
            },
            { role: "tool", tool_call_id: "call-1", content: "found" },
        ],
    }, fakeModels());

    const context = buildContext(parsed, 1234);
    assert.equal(context.tools, undefined);
    assert.deepEqual(context.messages[1].content, [{
        type: "toolCall",
        id: "call-1",
        name: "lookup",
        arguments: { q: "pi" },
    }]);
    assert.equal(context.messages[1].stopReason, "toolUse");
    assert.deepEqual(context.messages[2], {
        role: "toolResult",
        toolCallId: "call-1",
        toolName: "lookup",
        content: [{ type: "text", text: "found" }],
        isError: false,
        timestamp: 1234,
    });
});

test("malformed tool calls and tool results are rejected", () => {
    const withMessages = (messages) => () => parseRequest(
        { model: "provider-a/model-a", messages },
        fakeModels(),
        { warn() {} },
    );
    const calling = (toolCall) => [
        { role: "user", content: "q" },
        { role: "assistant", tool_calls: [toolCall] },
        { role: "tool", tool_call_id: "call-1", content: "x" },
    ];
    assert.throws(
        withMessages(calling({
            id: "call-1",
            function: { name: "lookup", arguments: "{oops" },
        })),
        /invalid arguments for tool lookup/,
    );
    assert.throws(
        withMessages(calling({
            id: "call-1",
            function: { name: "lookup", arguments: "[1]" },
        })),
        /arguments for tool lookup must be an object/,
    );
    assert.throws(
        withMessages(calling({ function: { name: "lookup" } })),
        /needs id and function.name/,
    );
    assert.throws(
        withMessages([
            { role: "user", content: "q" },
            { role: "tool", content: "x" },
        ]),
        /tool messages need tool_call_id/,
    );
});

test("tool declaration and tool_choice validation", () => {
    const warnings = [];
    const logger = { warn: (message) => warnings.push(message) };
    assert.deepEqual(parseTools([{
        type: "function",
        function: {
            name: "lookup",
            description: "look it up",
            parameters: { type: "object", properties: { q: {} } },
        },
    }, { function: { name: "bare" } }], undefined, logger), [
        {
            name: "lookup",
            description: "look it up",
            parameters: { type: "object", properties: { q: {} } },
        },
        {
            name: "bare",
            description: "",
            parameters: { type: "object", properties: {} },
        },
    ]);

    const tools = [{ type: "function", function: { name: "lookup" } }];
    assert.equal(parseTools(undefined, undefined, logger), undefined);
    assert.equal(parseTools([], undefined, logger), undefined);
    assert.equal(parseTools(tools, "none", logger), undefined);
    assert.equal(
        warnings.at(-1),
        "honoring tool_choice=none: sending no tools",
    );
    assert.equal(parseTools(tools, "auto", logger).length, 1);
    assert.equal(parseTools(tools, { type: "function" }, logger).length, 1);
    assert.match(warnings.at(-1), /ignoring unsupported tool_choice/);

    assert.throws(() => parseTools({}, undefined, logger), /must be an array/);
    assert.throws(
        () => parseTools([{ type: "custom", function: { name: "x" } }]),
        /unsupported tool type: custom/,
    );
    assert.throws(
        () => parseTools([{ type: "function", function: {} }]),
        /needs a non-empty function.name/,
    );
});

test("completion reports tool calls with stringified arguments", () => {
    const response = buildCompletion("model-a", assistantResult({
        content: [{
            type: "toolCall",
            id: "call-1",
            name: "lookup",
            arguments: { q: "pi" },
        }],
        stopReason: "toolUse",
    }));
    assert.deepEqual(response.choices[0], {
        index: 0,
        message: {
            role: "assistant",
            content: null,
            tool_calls: [{
                id: "call-1",
                type: "function",
                function: { name: "lookup", arguments: "{\"q\":\"pi\"}" },
            }],
        },
        finish_reason: "tool_calls",
    });
});

test("completion formats reasoning, finish reason, and token usage", () => {
    const response = buildCompletion("provider-a/model-a", assistantResult({
        content: [
            { type: "thinking", thinking: "step one" },
            { type: "text", text: "hello " },
            { type: "thinking", thinking: "step two" },
            { type: "text", text: "world" },
        ],
        stopReason: "length",
    }), { id: "fixed", created: 99 });
    assert.deepEqual(response, {
        id: "chatcmpl-fixed",
        object: "chat.completion",
        created: 99,
        model: "provider-a/model-a",
        choices: [{
            index: 0,
            message: {
                role: "assistant",
                content: "hello world",
                reasoning_content: "step one\n\nstep two",
            },
            finish_reason: "length",
        }],
        usage: { prompt_tokens: 7, completion_tokens: 3, total_tokens: 10 },
    });

    const failed = buildCompletion("model-a", assistantResult({
        stopReason: "error",
        errorMessage: "provider detail",
    }));
    assert.equal(failed.choices[0].finish_reason, "stop");
    assert.deepEqual(failed.choices[0].message, {
        role: "assistant",
        content: "answer",
    });
});

test("assistant stop on error warns but still returns 200", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({
            stopReason: "error",
            errorMessage: "provider detail",
        }),
    });
    const warnings = [];
    const logger = { warn: (message) => warnings.push(message), error() {} };
    const { baseUrl } = await startServer(models, t, logger);
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        body: JSON.stringify({
            model: "provider-a/model-a",
            messages: [{ role: "user", content: "question" }],
        }),
    });
    assert.equal(response.status, 200);
    assert.match(warnings[0], /provider detail/);
});

test("CLI parsing supports host, port, help, rejects bad arguments", () => {
    assert.deepEqual(parseArgs([]), {
        host: "127.0.0.1",
        port: 8742,
        help: false,
    });
    const arguments_ = ["--host=0.0.0.0", "--port", "9000", "--help"];
    assert.deepEqual(parseArgs(arguments_), {
        host: "0.0.0.0",
        port: 9000,
        help: true,
    });
    assert.throws(() => parseArgs(["--sessions", "2"]), UsageError);
    assert.throws(() => parseArgs(["--port", "nope"]), /invalid port/);
});

test("model runtime reads credentials from the auth file", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "pirouter-auth-"));
    const authPath = join(root, "auth.json");
    await writeFile(
        authPath,
        JSON.stringify({ openai: { type: "api_key", key: "secret" } }),
        { mode: 0o600 },
    );
    t.after(async () => {
        await rm(root, { recursive: true, force: true });
    });

    const runtime = await ModelRuntime.create({
        authPath,
        modelsPath: null,
        modelsStorePath: join(root, "models-store.json"),
        allowModelNetwork: false,
        refreshOnCreate: false,
    });

    assert.deepEqual(
        await runtime.listCredentials(),
        [{ providerId: "openai", type: "api_key" }],
    );
});

test("HTTP success calls completeSimple, returns OpenAI shape", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({
            content: [
                { type: "thinking", thinking: "reason" },
                { type: "text", text: "result" },
            ],
        }),
    });
    const { server, baseUrl } = await startServer(models, t);
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
            model: "provider-a/model-a",
            reasoning_effort: "medium",
            messages: [
                { role: "system", content: "rules" },
                { role: "user", content: "question" },
            ],
        }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.model, "provider-a/model-a");
    assert.equal(body.choices[0].message.content, "result");
    assert.equal(body.choices[0].message.reasoning_content, "reason");
    assert.deepEqual(body.usage, {
        prompt_tokens: 7,
        completion_tokens: 3,
        total_tokens: 10,
    });
    assert.match(body.id, /^chatcmpl-[0-9a-f]{32}$/);
    assert.equal(models.calls.length, 1);
    assert.equal(models.calls[0].model, MODEL);
    assert.deepEqual(models.calls[0].options, { reasoning: "medium" });
    assert.equal(models.calls[0].context.systemPrompt, "rules");
    assert.equal(server.listening, true);
});

test("HTTP round-trip forwards tools and returns tool calls", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({
            content: [
                { type: "text", text: "calling" },
                {
                    type: "toolCall",
                    id: "call-1",
                    name: "lookup",
                    arguments: { q: "pi" },
                },
            ],
            stopReason: "toolUse",
        }),
    });
    const { baseUrl } = await startServer(models, t);
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
            model: "provider-a/model-a",
            tools: [{ type: "function", function: { name: "lookup" } }],
            messages: [{ role: "user", content: "question" }],
        }),
    });
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.choices[0].finish_reason, "tool_calls");
    assert.equal(body.choices[0].message.content, "calling");
    assert.equal(body.choices[0].message.tool_calls[0].function.name, "lookup");
    assert.equal(models.calls[0].context.tools.length, 1);
});

test("parseRequest reads streaming flags", () => {
    const body = {
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    };
    const plain = parseRequest(body, fakeModels());
    assert.equal(plain.stream, false);
    assert.equal(plain.includeUsage, false);
    const streamed = parseRequest({
        ...body,
        stream: true,
        stream_options: { include_usage: true },
    }, fakeModels());
    assert.equal(streamed.stream, true);
    assert.equal(streamed.includeUsage, true);
});

const ENVELOPE = chunkEnvelope("provider-a/model-a", {
    id: "abc",
    created: 5,
});

test("buildChunks ignores events that carry no output", () => {
    assert.deepEqual(buildChunks(ENVELOPE, { type: "start" }, false), []);
    assert.deepEqual(
        buildChunks(ENVELOPE, { type: "toolcall_start" }, false),
        [],
    );
});

test("buildChunks maps text deltas to content deltas", () => {
    assert.deepEqual(
        buildChunks(ENVELOPE, { type: "text_delta", delta: "hi" }, false),
        [{
            id: "chatcmpl-abc",
            object: "chat.completion.chunk",
            created: 5,
            model: "provider-a/model-a",
            choices: [{
                index: 0,
                delta: { content: "hi" },
                finish_reason: null,
            }],
        }],
    );
});

test("buildChunks maps thinking deltas to reasoning deltas", () => {
    const [chunk] = buildChunks(
        ENVELOPE,
        { type: "thinking_delta", delta: "why" },
        false,
    );
    assert.deepEqual(chunk.choices[0].delta, { reasoning_content: "why" });
});

test("buildChunks numbers tool calls among tool calls only", () => {
    const [chunk] = buildChunks(ENVELOPE, {
        type: "toolcall_end",
        contentIndex: 2,
        toolCall: { id: "call-1", name: "lookup", arguments: { q: "pi" } },
        partial: {
            content: [
                { type: "text" },
                { type: "toolCall" },
                { type: "toolCall" },
            ],
        },
    }, false);
    assert.deepEqual(chunk.choices[0].delta.tool_calls, [{
        index: 1,
        id: "call-1",
        type: "function",
        function: { name: "lookup", arguments: "{\"q\":\"pi\"}" },
    }]);
});

test("buildChunks ends a done event with a finish reason and usage", () => {
    const [final, usage] = buildChunks(ENVELOPE, doneEvent(), true);
    assert.deepEqual(final.choices, [{
        index: 0,
        delta: {},
        finish_reason: "stop",
    }]);
    assert.deepEqual(usage.choices, []);
    assert.deepEqual(usage.usage, {
        prompt_tokens: 7,
        completion_tokens: 3,
        total_tokens: 10,
    });
    assert.equal(buildChunks(ENVELOPE, doneEvent(), false).length, 1);
});

// OpenAI has no failing finish_reason, so a failed turn also reads as "stop".
test("buildChunks ends an error event like a normal stop", () => {
    const chunks = buildChunks(ENVELOPE, {
        type: "error",
        reason: "error",
        error: assistantResult({ stopReason: "error" }),
    }, false);
    assert.equal(chunks.length, 1);
    assert.equal(chunks[0].choices[0].finish_reason, "stop");
});

test("HTTP streaming sends SSE chunks and [DONE]", async (t) => {
    const models = fakeModels({
        stream: async function* () {
            yield { type: "text_delta", delta: "he" };
            yield { type: "text_delta", delta: "llo" };
            yield doneEvent({ stopReason: "toolUse", content: [
                { type: "toolCall", id: "call-1", name: "lookup" },
            ] });
        },
    });
    const { baseUrl } = await startServer(models, t);
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
            model: "provider-a/model-a",
            stream: true,
            stream_options: { include_usage: true },
            messages: [{ role: "user", content: "question" }],
        }),
    });
    assert.equal(response.status, 200);
    assert.match(response.headers.get("content-type"), /text\/event-stream/);

    const frames = await readEventStream(response);
    assert.equal(frames.at(-1), "[DONE]");
    const chunks = frames.slice(0, -1).map((frame) => JSON.parse(frame));
    assert.deepEqual(chunks[0].choices[0].delta, {
        role: "assistant",
        content: "",
    });
    assert.deepEqual(
        chunks.slice(1, 3).map((chunk) => chunk.choices[0].delta.content),
        ["he", "llo"],
    );
    assert.equal(chunks[3].choices[0].finish_reason, "tool_calls");
    assert.deepEqual(chunks[4].usage, {
        prompt_tokens: 7,
        completion_tokens: 3,
        total_tokens: 10,
    });
    assert.equal(chunks.every((chunk) => (
        chunk.object === "chat.completion.chunk"
    )), true);
    const { signal } = models.calls[0].options;
    assert.equal(signal instanceof globalThis.AbortSignal, true);
});

test("HTTP streaming omits usage unless requested", async (t) => {
    const models = fakeModels({
        stream: async function* () {
            yield doneEvent();
        },
    });
    const { baseUrl } = await startServer(models, t);
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
            model: "provider-a/model-a",
            stream: true,
            messages: [{ role: "user", content: "question" }],
        }),
    });
    const frames = await readEventStream(response);
    const chunks = frames.slice(0, -1).map((frame) => JSON.parse(frame));
    assert.equal(chunks.length, 2);
    assert.equal(chunks[1].choices[0].finish_reason, "stop");
});

test("HTTP streaming ends the stream when the model fails", async (t) => {
    const errors = [];
    const models = fakeModels({
        stream: async function* () {
            yield { type: "text_delta", delta: "partial" };
            throw new Error("provider unavailable");
        },
    });
    const { baseUrl } = await startServer(models, t, {
        warn() {},
        error(...args) { errors.push(args); },
    });
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
            model: "provider-a/model-a",
            stream: true,
            messages: [{ role: "user", content: "question" }],
        }),
    });
    assert.equal(response.status, 200);
    const frames = await readEventStream(response);
    assert.equal(frames.at(-1), "[DONE]");
    const last = JSON.parse(frames.at(-2));
    assert.equal(last.choices[0].finish_reason, "stop");
    assert.equal(errors.length, 1);
});

test("HTTP streaming aborts the model when the client leaves", async (t) => {
    let released;
    const blocked = new Promise((resolve) => { released = resolve; });
    const models = fakeModels({
        stream: async function* () {
            yield { type: "text_delta", delta: "partial" };
            await blocked;
            yield doneEvent();
        },
    });
    const { baseUrl } = await startServer(models, t);
    const controller = new globalThis.AbortController();
    const response = await globalThis.fetch(`${baseUrl}/chat/completions`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
            model: "provider-a/model-a",
            stream: true,
            messages: [{ role: "user", content: "question" }],
        }),
    });
    const reader = response.body.getReader();
    await reader.read();
    controller.abort();
    const signal = models.calls[0].options.signal;
    await new Promise((resolve) => {
        signal.addEventListener("abort", resolve, { once: true });
    });
    released();
    assert.equal(signal.aborted, true);
});

test("HTTP errors cover bad JSON, path, model, provider failure", async (t) => {
    const duplicate = { ...MODEL, provider: "provider-b" };
    const models = fakeModels({
        models: [MODEL, duplicate],
        complete: async () => {
            throw new Error("provider unavailable");
        },
    });
    const { baseUrl } = await startServer(models, t, { error() {} });

    const messages = [{ role: "user", content: "x" }];
    const cases = [
        ["/wrong", "{}", 404, "not found"],
        ["/chat/completions", "{", 400, "invalid JSON"],
        [
            "/chat/completions",
            JSON.stringify({ messages }),
            400,
            "model must",
        ],
        [
            "/chat/completions",
            JSON.stringify({ model: "model-a", messages }),
            400,
            "ambiguous model",
        ],
        [
            "/chat/completions",
            JSON.stringify({ model: "provider-a/missing", messages }),
            400,
            "unknown model",
        ],
        [
            "/chat/completions",
            JSON.stringify({ model: "provider-a/model-a", messages }),
            500,
            "provider unavailable",
        ],
    ];
    for (const [path, requestBody, status, message] of cases) {
        const response = await globalThis.fetch(`${baseUrl}${path}`, {
            method: "POST",
            body: requestBody,
        });
        assert.equal(response.status, status);
        const body = await response.json();
        assert.match(body.error.message, new RegExp(message));
        assert.equal(body.error.code, status);
        const expectedType = status === 500
            ? "server_error"
            : "invalid_request_error";
        assert.equal(body.error.type, expectedType);
    }
});

test("launcher forwards arguments without invoking npm", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "pirouter-launcher-"));
    const marker = join(root, "npm-called");
    const npm = join(root, "npm");
    const fakeNpm =
        `#!/usr/bin/env sh\nprintf called > ${JSON.stringify(marker)}\nexit 99\n`;
    await writeFile(npm, fakeNpm);
    await chmod(npm, 0o700);
    t.after(async () => {
        await rm(root, { recursive: true, force: true });
    });

    const project = fileURLToPath(new URL(".", import.meta.url));
    const arguments_ = [
        "--host",
        "forwarded.example",
        "--port",
        "1234",
        "--help",
    ];
    const env = {
        ...process.env,
        PATH: `${root}:${process.env.PATH}`,
    };
    const result = await run(
        join(project, "pi-router"),
        arguments_,
        env,
    );
    assert.equal(result.code, 0, result.stderr);
    assert.equal(
        result.stdout,
        "Usage: pirouter [--host HOST] [--port PORT]\n",
    );
    await assert.rejects(stat(marker), { code: "ENOENT" });
});

async function readEventStream(response) {
    const text = await response.text();
    return text
        .split("\n\n")
        .filter((frame) => frame.length > 0)
        .map((frame) => frame.replace(/^data: /, ""));
}

async function startServer(models, t, logger = { warn() {}, error() {} }) {
    const server = createChatServer({ models, logger });
    await new Promise((resolve, reject) => {
        server.once("error", reject);
        server.listen(0, "127.0.0.1", resolve);
    });
    t.after(() => new Promise((resolve) => server.close(resolve)));
    const { port } = server.address();
    return { server, baseUrl: `http://127.0.0.1:${port}` };
}

function run(command, args, env) {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, { env });
        let stdout = "";
        let stderr = "";
        child.stdout.on("data", (chunk) => { stdout += chunk; });
        child.stderr.on("data", (chunk) => { stderr += chunk; });
        child.on("error", reject);
        child.on("close", (code, signal) => {
            resolve({ code, signal, stdout, stderr });
        });
    });
}
