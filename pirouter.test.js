import assert from "node:assert/strict";
import { Buffer } from "node:buffer";
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
    CHAT_COMPLETIONS_PATH,
    UsageError,
    buildChunks,
    buildCompletion,
    buildContext,
    chunkEnvelope,
    createChatServer,
    extractText,
    parseArgs,
    parseReasoningOption,
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

const OTHER_MODEL = {
    id: "model-b",
    provider: "provider-b",
    api: "anthropic-messages",
};

const CHAT_REQUEST = { requestedModel: "provider-a/model-a", model: MODEL };

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
        hasConfiguredAuth() {
            return false;
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
        usage: piUsage(),
        stopReason: "stop",
        timestamp: 1,
        ...overrides,
    };
}

function piUsage({ cost: costOverrides = {}, ...usageOverrides } = {}) {
    return {
        input: 7,
        output: 5,
        cacheRead: 11,
        cacheWrite: 13,
        cacheWrite1h: 4,
        reasoning: 2,
        totalTokens: 99,
        ...usageOverrides,
        cost: {
            input: 1.25,
            output: 3.25,
            cacheRead: 0.5,
            cacheWrite: 0.75,
            total: 9.5,
            ...costOverrides,
        },
    };
}

const EXPECTED_USAGE = {
    prompt_tokens: 31,
    completion_tokens: 5,
    total_tokens: 99,
    prompt_tokens_details: {
        cached_tokens: 11,
        cache_write_tokens: 13,
    },
    cost: 9.5,
    cost_details: {
        upstream_inference_prompt_cost: 2.5,
        upstream_inference_completions_cost: 3.25,
        upstream_inference_cost: 9.5,
    },
    completion_tokens_details: { reasoning_tokens: 2 },
};

// pi-router stamps a replayed turn it cannot attribute with this identity.
const UNVERIFIED_MODEL = {
    api: "pirouter-unverified",
    provider: "pirouter-unverified",
    model: "pirouter-unverified",
};

const REPLAYED_USAGE = {
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
};

test("content and message normalization keep text and drop nothing", () => {
    assert.equal(extractText("plain"), "plain");
    assert.equal(extractText([
        { type: "text", text: "one" },
        { type: "text", text: "two" },
    ]), "onetwo");
    assert.equal(extractText(null), "");
    assert.throws(
        () => extractText([{ type: "image_url", image_url: "x" }]),
        /unsupported message content part: image_url/,
    );

    const { systemPrompt, conversation } = splitMessages([
        { role: "system", content: "system" },
        { role: "developer", content: [{ type: "text", text: "developer" }] },
        { role: "assistant", content: "old answer" },
        { role: "user", content: [{ type: "text", text: "new question" }] },
    ], MODEL);
    assert.equal(systemPrompt, "system\n\ndeveloper");
    assert.deepEqual(conversation[0].content, [
        { type: "text", text: "old answer" },
    ]);
    assert.equal(conversation[0].stopReason, "stop");
    assert.deepEqual(conversation[1], {
        role: "user",
        content: "new question",
    });
});

test("model resolution accepts the forms Pi accepts", () => {
    const duplicate = { ...MODEL, provider: "provider-b" };
    const models = fakeModels({ models: [MODEL, duplicate] });
    assert.equal(resolveModel(fakeModels(), "provider-a/model-a"), MODEL);
    assert.equal(resolveModel(fakeModels(), "model-a"), MODEL);
    assert.equal(resolveModel(fakeModels(), "provider-a/model"), MODEL);
    assert.throws(() => resolveModel(models, "model-a"), /ambiguous/);
    assert.throws(() => resolveModel(fakeModels(), "missing"), /not found/);
    assert.throws(
        () => resolveModel(fakeModels(), ""),
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
        reasoning: { effort: "high" },
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

    assert.equal(parseReasoningOption(undefined), undefined);
    assert.equal(parseReasoningOption({ effort: "none" }), undefined);
    assert.equal(parseReasoningOption({ effort: "minimal" }), "minimal");
    assert.equal(parseReasoningOption({ enabled: true }), "medium");
    assert.equal(parseReasoningOption({ enabled: false }), undefined);
    assert.equal(
        parseReasoningOption({ enabled: true, max_tokens: 100 }, logger),
        "medium",
    );
    assert.equal(warnings.at(-1), "ignoring unsupported reasoning member: "
        + "max_tokens");
    assert.equal(parseReasoningOption({ effort: "max" }, logger), undefined);
    assert.match(warnings.at(-1), /ignoring unknown reasoning effort/);

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
        ...UNVERIFIED_MODEL,
        usage: REPLAYED_USAGE,
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
    const response = buildCompletion(CHAT_REQUEST, assistantResult({
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
    const response = buildCompletion(CHAT_REQUEST, assistantResult({
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
                reasoning: "step one\n\nstep two",
                reasoning_details: [
                    {
                        type: "reasoning.text",
                        text: "step one",
                        format: "unknown",
                        id: null,
                        index: 0,
                    },
                    {
                        type: "reasoning.text",
                        text: "step two",
                        format: "unknown",
                        id: null,
                        index: 1,
                    },
                ],
            },
            finish_reason: "length",
        }],
        usage: EXPECTED_USAGE,
    });
});

test("completion names the provider family that reasoned", () => {
    const response = buildCompletion(
        { requestedModel: "provider-b/model-b", model: OTHER_MODEL },
        assistantResult({
            content: [{ type: "thinking", thinking: "why" }],
        }),
    );
    assert.equal(
        response.choices[0].message.reasoning_details[0].format,
        "anthropic-claude-v1",
    );
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

test("HTTP serves only the OpenRouter path and method", async (t) => {
    const { baseUrl } = await startServer(fakeModels(), t);
    const body = JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    });
    const rejected = [
        ["/chat/completions", "POST"],
        ["/api/v1/chat/completions", "GET"],
    ];
    for (const [path, method] of rejected) {
        const response = await globalThis.fetch(`${baseUrl}${path}`, {
            method,
            body: method === "POST" ? body : undefined,
        });
        assert.equal(response.status, 404);
    }
    const accepted = await post(baseUrl, body);
    assert.equal(accepted.status, 200);
});

test("HTTP accepts any credential and none", async (t) => {
    const { baseUrl } = await startServer(fakeModels(), t);
    const body = JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    });
    const withKey = await post(baseUrl, body, {
        authorization: "Bearer placeholder",
    });
    assert.equal(withKey.status, 200);
    assert.equal((await post(baseUrl, body)).status, 200);
});

test("HTTP selects models by full, short, and bad names", async (t) => {
    const duplicate = { ...MODEL, provider: "provider-b" };
    const models = fakeModels({ models: [MODEL, duplicate] });
    const { baseUrl } = await startServer(models, t);
    const messages = [{ role: "user", content: "question" }];
    const ask = (model) => post(
        baseUrl,
        JSON.stringify({ model, messages }),
    );

    const full = await ask("provider-a/model-a");
    assert.equal(full.status, 200);
    assert.equal((await full.json()).model, "provider-a/model-a");
    assert.equal(models.calls[0].model, MODEL);

    const short = await ask("provider-b/model");
    assert.equal(short.status, 200);
    assert.equal(models.calls[1].model, duplicate);

    const ambiguous = await ask("model-a");
    assert.equal(ambiguous.status, 400);
    assert.match((await ambiguous.json()).error.message, /ambiguous/);

    const unknown = await ask("nonesuch");
    assert.equal(unknown.status, 400);
    assert.match((await unknown.json()).error.message, /not found/);
});

test("HTTP maps the reasoning object to Pi thinking levels", async (t) => {
    const warnings = [];
    const models = fakeModels();
    const { baseUrl } = await startServer(models, t, {
        warn: (message) => warnings.push(message),
        error() {},
    });
    const messages = [{ role: "user", content: "question" }];
    const cases = [
        [{ effort: "high" }, { reasoning: "high" }],
        [{ effort: "none" }, {}],
        [{ enabled: true }, { reasoning: "medium" }],
        [{ enabled: false }, {}],
        [{ enabled: true, exclude: true, max_tokens: 100 }, {
            reasoning: "medium",
        }],
        [undefined, {}],
    ];
    for (const [reasoning, expected] of cases) {
        const body = { model: "provider-a/model-a", messages };
        if (reasoning !== undefined) body.reasoning = reasoning;
        const response = await post(baseUrl, JSON.stringify(body));
        assert.equal(response.status, 200);
        assert.deepEqual(models.calls.at(-1).options, expected);
    }
    assert.deepEqual(warnings, [
        "ignoring unsupported reasoning member: exclude",
        "ignoring unsupported reasoning member: max_tokens",
    ]);
});

test("HTTP replays reasoning to the model that produced it", async (t) => {
    const models = fakeModels({
        models: [MODEL, OTHER_MODEL],
        complete: async () => assistantResult({
            content: [
                {
                    type: "thinking",
                    thinking: "why",
                    thinkingSignature: "reasoning_content",
                },
                {
                    type: "thinking",
                    thinking: "[Reasoning redacted]",
                    thinkingSignature: "blob",
                    redacted: true,
                },
                { type: "text", text: "answer" },
            ],
        }),
    });
    const { baseUrl } = await startServer(models, t);
    const first = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    }));
    const message = (await first.json()).choices[0].message;
    assert.deepEqual(message.reasoning_details, [
        {
            type: "reasoning.text",
            text: "why",
            signature: message.reasoning_details[0].signature,
            format: "unknown",
            id: null,
            index: 0,
        },
        {
            type: "reasoning.encrypted",
            data: message.reasoning_details[1].data,
            format: "unknown",
            id: null,
            index: 1,
        },
    ]);
    // The signature is opaque: it is neither the value Pi stored nor a value
    // an OpenAI-compatible route could use as a bare request field name.
    const { signature } = message.reasoning_details[0];
    assert.notEqual(signature, "reasoning_content");
    assert.equal(/^[A-Za-z_][A-Za-z0-9_]*$/.test(signature), false);

    const replay = (model) => post(baseUrl, JSON.stringify({
        model,
        messages: [
            { role: "user", content: "question" },
            {
                role: "assistant",
                content: message.content,
                reasoning_details: message.reasoning_details,
            },
            { role: "user", content: "again" },
        ],
    }));

    await replay("provider-a/model-a");
    const replayed = models.calls.at(-1).context.messages[1];
    assert.deepEqual(replayed.content, [
        {
            type: "thinking",
            thinking: "why",
            thinkingSignature: "reasoning_content",
        },
        {
            type: "thinking",
            thinking: "[Reasoning redacted]",
            thinkingSignature: "blob",
            redacted: true,
        },
        { type: "text", text: "answer" },
    ]);
    assert.equal(replayed.model, "model-a");
    assert.equal(replayed.provider, "provider-a");

    await replay("provider-b/model-b");
    const switched = models.calls.at(-1).context.messages[1];
    assert.deepEqual(switched.content, [
        { type: "thinking", thinking: "why" },
        { type: "text", text: "answer" },
    ]);
    assert.equal(switched.provider, "provider-a");
    assert.equal(switched.api, "test-api");
});

test("HTTP strips unverifiable signatures and marks the turn", async (t) => {
    const unverifiable = [
        "anthropic-signature",
        "pirouter-v1.!!not-base64!!",
        `pirouter-v1.${Buffer.from(JSON.stringify({
            api: "test-api",
            provider: "provider-a",
            id: "model-a",
        })).toString("base64url")}`,
    ];
    const models = fakeModels();
    const { baseUrl } = await startServer(models, t);
    for (const signature of unverifiable) {
        const response = await post(baseUrl, JSON.stringify({
            model: "provider-a/model-a",
            messages: [
                { role: "user", content: "question" },
                {
                    role: "assistant",
                    content: "answer",
                    reasoning_details: [
                        { type: "reasoning.text", text: "why", signature },
                        { type: "reasoning.encrypted", data: signature },
                    ],
                },
                { role: "user", content: "again" },
            ],
        }));
        assert.equal(response.status, 200);
        const replayed = models.calls.at(-1).context.messages[1];
        assert.deepEqual(replayed.content, [
            { type: "thinking", thinking: "why" },
            { type: "text", text: "answer" },
        ]);
        // No real model has this identity, so Pi treats the reasoning as
        // foreign instead of as the selected model's own.
        assert.deepEqual({
            api: replayed.api,
            provider: replayed.provider,
            model: replayed.model,
        }, UNVERIFIED_MODEL);
    }
});

test("HTTP keeps signatures of one model in a mixed turn", async (t) => {
    const models = fakeModels({ models: [MODEL, OTHER_MODEL] });
    const { baseUrl } = await startServer(models, t);
    const signed = (model, signature) => {
        const request = { requestedModel: model.id, model };
        return buildCompletion(request, assistantResult({
            content: [{
                type: "thinking",
                thinking: `by ${model.id}`,
                thinkingSignature: signature,
            }],
        })).choices[0].message.reasoning_details[0];
    };

    const ask = (model) => post(baseUrl, JSON.stringify({
        model,
        messages: [
            { role: "user", content: "question" },
            {
                role: "assistant",
                content: "answer",
                reasoning_details: [
                    signed(MODEL, "signature-a"),
                    signed(OTHER_MODEL, "signature-b"),
                ],
            },
            { role: "user", content: "again" },
        ],
    }));

    // The first entry that names its model is the turn's origin, so only its
    // signature may return, and only to that same model.
    const toOrigin = await ask("provider-a/model-a");
    assert.equal(toOrigin.status, 200);
    const replayed = models.calls.at(-1).context.messages[1];
    assert.deepEqual(replayed.content, [
        {
            type: "thinking",
            thinking: "by model-a",
            thinkingSignature: "signature-a",
        },
        { type: "thinking", thinking: "by model-b" },
        { type: "text", text: "answer" },
    ]);
    assert.equal(replayed.model, MODEL.id);
    assert.equal(replayed.provider, MODEL.provider);

    const toOther = await ask("provider-b/model-b");
    assert.equal(toOther.status, 200);
    const switched = models.calls.at(-1).context.messages[1];
    assert.deepEqual(switched.content, [
        { type: "thinking", thinking: "by model-a" },
        { type: "thinking", thinking: "by model-b" },
        { type: "text", text: "answer" },
    ]);
    assert.equal(switched.model, MODEL.id);
});

test("HTTP success returns the OpenRouter shape", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({
            content: [
                { type: "thinking", thinking: "reason" },
                { type: "text", text: "result" },
            ],
        }),
    });
    const { baseUrl } = await startServer(models, t);
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        reasoning: { effort: "medium" },
        messages: [
            { role: "system", content: "rules" },
            { role: "user", content: "question" },
        ],
    }));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.model, "provider-a/model-a");
    assert.equal(body.choices[0].message.content, "result");
    assert.equal(body.choices[0].message.reasoning, "reason");
    assert.deepEqual(body.usage, EXPECTED_USAGE);
    assert.match(body.id, /^chatcmpl-[0-9a-f]{32}$/);
    assert.equal(models.calls.length, 1);
    assert.equal(models.calls[0].model, MODEL);
    assert.deepEqual(models.calls[0].options, { reasoning: "medium" });
    assert.equal(models.calls[0].context.systemPrompt, "rules");
});

test("HTTP usage omits unknown reasoning", async (t) => {
    const usage = piUsage();
    delete usage.reasoning;
    const models = fakeModels({
        complete: async () => assistantResult({ usage }),
    });
    const { baseUrl } = await startServer(models, t);
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    }));

    assert.equal(
        Object.hasOwn(
            (await response.json()).usage,
            "completion_tokens_details",
        ),
        false,
    );
});

test("HTTP usage keeps explicit zero reasoning", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({
            usage: piUsage({ reasoning: 0 }),
        }),
    });
    const { baseUrl } = await startServer(models, t);
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    }));

    assert.deepEqual(
        (await response.json()).usage.completion_tokens_details,
        { reasoning_tokens: 0 },
    );
});

test("HTTP buffered responses retain all-zero usage", async (t) => {
    const usage = piUsage({
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        reasoning: undefined,
        totalTokens: 0,
        cost: {
            input: 0,
            output: 0,
            cacheRead: 0,
            cacheWrite: 0,
            total: 0,
        },
    });
    const models = fakeModels({
        complete: async () => assistantResult({ usage }),
    });
    const { baseUrl } = await startServer(models, t);
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    }));

    assert.deepEqual((await response.json()).usage, {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0,
        prompt_tokens_details: {
            cached_tokens: 0,
            cache_write_tokens: 0,
        },
        cost: 0,
        cost_details: {
            upstream_inference_prompt_cost: 0,
            upstream_inference_completions_cost: 0,
            upstream_inference_cost: 0,
        },
    });
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
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        tools: [{ type: "function", function: { name: "lookup" } }],
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.choices[0].finish_reason, "tool_calls");
    assert.equal(body.choices[0].message.content, "calling");
    assert.equal(body.choices[0].message.tool_calls[0].function.name, "lookup");
    assert.equal(models.calls[0].context.tools.length, 1);
});

test("HTTP sends no tools when the client asks for none", async (t) => {
    const warnings = [];
    const models = fakeModels();
    const { baseUrl } = await startServer(models, t, {
        warn: (message) => warnings.push(message),
        error() {},
    });
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        tools: [{ type: "function", function: { name: "lookup" } }],
        tool_choice: "none",
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 200);
    assert.equal(models.calls[0].context.tools, undefined);
    assert.equal(
        warnings.includes("honoring tool_choice=none: sending no tools"),
        true,
    );
});

test("HTTP offers tools when a tool is demanded", async (t) => {
    const warnings = [];
    const models = fakeModels();
    const { baseUrl } = await startServer(models, t, {
        warn: (message) => warnings.push(message),
        error() {},
    });
    const demands = ["required", { type: "function", function: {
        name: "lookup",
    } }];
    for (const toolChoice of demands) {
        const response = await post(baseUrl, JSON.stringify({
            model: "provider-a/model-a",
            tools: [{ type: "function", function: { name: "lookup" } }],
            tool_choice: toolChoice,
            messages: [{ role: "user", content: "question" }],
        }));
        assert.equal(response.status, 200);
        assert.equal(models.calls.at(-1).context.tools.length, 1);
    }
    assert.equal(
        warnings.filter((warning) => (
            warning.startsWith("ignoring unsupported tool_choice")
        )).length,
        2,
    );
});

test("HTTP tolerates fields it cannot honor", async (t) => {
    const warnings = [];
    const models = fakeModels();
    const { baseUrl } = await startServer(models, t, {
        warn: (message) => warnings.push(message),
        error() {},
    });
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        temperature: 0.4,
        max_tokens: 100,
        stop: ["\n"],
        provider: { order: ["anthropic"] },
        models: ["other/model"],
        unknown_extra: true,
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 200);
    assert.deepEqual(warnings, [
        "ignoring unsupported field: temperature",
        "ignoring unsupported field: max_tokens",
        "ignoring unsupported field: stop",
        "ignoring unsupported field: provider",
        "ignoring unsupported field: models",
        "ignoring unsupported field: unknown_extra",
    ]);
});

test("HTTP rejects images and constrained output formats", async (t) => {
    const { baseUrl } = await startServer(fakeModels(), t);
    const rejected = [
        [
            {
                model: "provider-a/model-a",
                messages: [{
                    role: "user",
                    content: [
                        { type: "text", text: "what is this?" },
                        { type: "image_url", image_url: { url: "data:..." } },
                    ],
                }],
            },
            /unsupported message content part: image_url/,
        ],
        [
            {
                model: "provider-a/model-a",
                response_format: {
                    type: "json_schema",
                    json_schema: { name: "answer", schema: {} },
                },
                messages: [{ role: "user", content: "question" }],
            },
            /unsupported response_format: json_schema/,
        ],
        [
            {
                model: "provider-a/model-a",
                response_format: { type: "json_object" },
                messages: [{ role: "user", content: "question" }],
            },
            /unsupported response_format: json_object/,
        ],
    ];
    for (const [body, expected] of rejected) {
        const response = await post(baseUrl, JSON.stringify(body));
        assert.equal(response.status, 400);
        const error = (await response.json()).error;
        assert.match(error.message, expected);
        assert.equal(error.type, "invalid_request_error");
    }
});

test("HTTP reports a failed turn as a provider error", async (t) => {
    const errors = [];
    const failures = [
        ["429 Too Many Requests: slow down", 429],
        ["provider unavailable", 500],
        // A leading number outside the HTTP status range is not a status.
        ["200 OK but truncated", 500],
    ];
    for (const [errorMessage, status] of failures) {
        const models = fakeModels({
            complete: async () => assistantResult({
                stopReason: "error",
                errorMessage,
            }),
        });
        const { baseUrl } = await startServer(models, t, {
            warn() {},
            error(...args) { errors.push(args); },
        });
        const response = await post(baseUrl, JSON.stringify({
            model: "provider-a/model-a",
            messages: [{ role: "user", content: "question" }],
        }));
        assert.equal(response.status, status);
        const error = (await response.json()).error;
        assert.equal(error.message, errorMessage);
        assert.equal(error.code, status);
        assert.equal(error.type, "provider_error");
    }
    assert.equal(errors.length, failures.length);
});

test("HTTP reports an aborted turn as a provider error", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({ stopReason: "aborted" }),
    });
    const { baseUrl } = await startServer(models, t, {
        warn() {},
        error() {},
    });
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 500);
    const error = (await response.json()).error;
    assert.equal(error.message, "model stopped: aborted");
    assert.equal(error.type, "provider_error");
});

test("HTTP reports blocked content as content_filter", async (t) => {
    const models = fakeModels({
        complete: async () => assistantResult({
            content: [{ type: "text", text: "" }],
            stopReason: "error",
            rawStopReason: "content_filter",
            errorMessage: "Provider finish_reason: content_filter",
        }),
    });
    const { baseUrl } = await startServer(models, t);
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 200);
    const body = await response.json();
    assert.equal(body.choices[0].finish_reason, "content_filter");
});

test("parseRequest reads streaming flags", () => {
    const body = {
        model: "provider-a/model-a",
        messages: [{ role: "user", content: "question" }],
    };
    const plain = parseRequest(body, fakeModels());
    assert.equal(plain.stream, false);
    const streamed = parseRequest({ ...body, stream: true }, fakeModels());
    assert.equal(streamed.stream, true);
});

const ENVELOPE = chunkEnvelope("provider-a/model-a", {
    id: "abc",
    created: 5,
});

test("buildChunks ignores events that carry no output", () => {
    assert.deepEqual(buildChunks(ENVELOPE, { type: "start" }, MODEL), []);
    assert.deepEqual(
        buildChunks(ENVELOPE, { type: "toolcall_start" }, MODEL),
        [],
    );
    assert.deepEqual(
        buildChunks(ENVELOPE, {
            type: "thinking_end",
            contentIndex: 3,
            partial: { content: [] },
        }, MODEL),
        [],
    );
});

test("buildChunks maps text deltas to content deltas", () => {
    assert.deepEqual(
        buildChunks(ENVELOPE, { type: "text_delta", delta: "hi" }, MODEL),
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
        MODEL,
    );
    assert.deepEqual(chunk.choices[0].delta, { reasoning: "why" });
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
    }, MODEL);
    assert.deepEqual(chunk.choices[0].delta.tool_calls, [{
        index: 1,
        id: "call-1",
        type: "function",
        function: { name: "lookup", arguments: "{\"q\":\"pi\"}" },
    }]);
});

test("buildChunks ends a done event with a finish reason and usage", () => {
    const [final, usage] = buildChunks(ENVELOPE, doneEvent(), MODEL);
    assert.deepEqual(final.choices, [{
        index: 0,
        delta: {},
        finish_reason: "stop",
    }]);
    assert.deepEqual(usage.choices, [{
        index: 0,
        delta: {},
        finish_reason: null,
    }]);
    assert.deepEqual(usage.usage, EXPECTED_USAGE);
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
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        stream: true,
        stream_options: { include_usage: true },
        messages: [{ role: "user", content: "question" }],
    }));
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
        chunks
            .map((chunk) => chunk.choices[0].delta.content)
            .filter((content) => content),
        ["he", "llo"],
    );
    assert.deepEqual(
        chunks
            .map((chunk) => chunk.choices[0].finish_reason)
            .filter((reason) => reason !== null),
        ["tool_calls"],
    );
    const usageChunks = chunks.filter((chunk) => chunk.usage !== undefined);
    assert.equal(usageChunks.length, 1);
    assert.deepEqual(usageChunks[0].choices, [{
        index: 0,
        delta: {},
        finish_reason: null,
    }]);
    assert.equal(chunks.every((chunk) => (
        chunk.object === "chat.completion.chunk"
    )), true);
    const { signal } = models.calls[0].options;
    assert.equal(signal instanceof globalThis.AbortSignal, true);
});

test("HTTP streaming always sends usage after the finish chunk", async (t) => {
    const models = fakeModels({
        stream: async function* () {
            yield doneEvent();
        },
    });
    const { baseUrl } = await startServer(models, t);
    const streamOptions = [undefined, { include_usage: false }];
    for (const options of streamOptions) {
        const body = {
            model: "provider-a/model-a",
            stream: true,
            messages: [{ role: "user", content: "question" }],
        };
        if (options !== undefined) body.stream_options = options;
        const response = await post(baseUrl, JSON.stringify(body));
        const frames = await readEventStream(response);
        assert.equal(frames.at(-1), "[DONE]");
        const chunks = frames.slice(0, -1).map((frame) => JSON.parse(frame));
        const usageChunks = chunks.filter((chunk) => (
            chunk.usage !== undefined
        ));
        assert.equal(usageChunks.length, 1);
        assert.equal(usageChunks.at(-1), chunks.at(-1));
        assert.deepEqual(usageChunks[0].choices[0].delta, {});
        assert.deepEqual(usageChunks[0].usage, EXPECTED_USAGE);
        assert.deepEqual(
            chunks
                .map((chunk) => chunk.choices[0].finish_reason)
                .filter((reason) => reason !== null),
            ["stop"],
        );
    }
});

test("HTTP streams reasoning and replays its signature", async (t) => {
    const models = fakeModels({
        stream: async function* () {
            yield { type: "thinking_delta", delta: "why" };
            yield {
                type: "thinking_end",
                contentIndex: 0,
                partial: {
                    content: [{
                        type: "thinking",
                        thinking: "why",
                        thinkingSignature: "reasoning_content",
                    }],
                },
            };
            yield doneEvent();
        },
    });
    const { baseUrl } = await startServer(models, t);
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        stream: true,
        messages: [{ role: "user", content: "question" }],
    }));
    const chunks = (await readEventStream(response))
        .slice(0, -1)
        .map((frame) => JSON.parse(frame));
    const deltas = chunks.map((chunk) => chunk.choices[0].delta);
    assert.deepEqual(
        deltas.map((delta) => delta.reasoning).filter((text) => text),
        ["why"],
    );
    const [detail] = deltas
        .flatMap((delta) => delta.reasoning_details ?? []);
    assert.equal(detail.type, "reasoning.text");
    assert.equal(detail.text, "why");
    assert.equal(detail.index, 0);
    assert.equal(detail.format, "unknown");
    assert.equal(detail.id, null);

    const replay = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        messages: [
            { role: "user", content: "question" },
            {
                role: "assistant",
                content: "answer",
                reasoning_details: [detail],
            },
            { role: "user", content: "again" },
        ],
    }));
    assert.equal(replay.status, 200);
    assert.deepEqual(models.calls.at(-1).context.messages[1].content, [
        {
            type: "thinking",
            thinking: "why",
            thinkingSignature: "reasoning_content",
        },
        { type: "text", text: "answer" },
    ]);
});

test("HTTP streaming reports a failure before it starts", async (t) => {
    const errors = [];
    const models = fakeModels({
        // eslint-disable-next-line require-yield
        stream: async function* () {
            throw new Error("503 Service Unavailable");
        },
    });
    const { baseUrl } = await startServer(models, t, {
        warn() {},
        error(...args) { errors.push(args); },
    });
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        stream: true,
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 503);
    assert.match(response.headers.get("content-type"), /application\/json/);
    const error = (await response.json()).error;
    assert.equal(error.message, "503 Service Unavailable");
    assert.equal(error.type, "provider_error");
    assert.equal(errors.length, 1);
});

test("HTTP streaming reports a failure as a stream error event", async (t) => {
    const errors = [];
    const models = fakeModels({
        stream: async function* () {
            yield { type: "text_delta", delta: "partial" };
            yield {
                type: "error",
                reason: "error",
                error: assistantResult({
                    stopReason: "error",
                    errorMessage: "429 Too Many Requests",
                }),
            };
        },
    });
    const { baseUrl } = await startServer(models, t, {
        warn() {},
        error(...args) { errors.push(args); },
    });
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        stream: true,
        messages: [{ role: "user", content: "question" }],
    }));
    assert.equal(response.status, 200);
    const frames = await readEventStream(response);
    assert.equal(frames.at(-1), "[DONE]");
    const chunks = frames.slice(0, -1).map((frame) => JSON.parse(frame));
    assert.deepEqual(chunks.at(-1), {
        error: {
            message: "429 Too Many Requests",
            type: "provider_error",
            code: 429,
        },
    });
    assert.equal(chunks.some((chunk) => chunk.usage !== undefined), false);
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
    const response = await post(baseUrl, JSON.stringify({
        model: "provider-a/model-a",
        stream: true,
        messages: [{ role: "user", content: "question" }],
    }), {}, controller.signal);
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

test("HTTP errors cover bad JSON and provider failure", async (t) => {
    const models = fakeModels({
        complete: async () => {
            throw new Error("provider unavailable");
        },
    });
    const { baseUrl } = await startServer(models, t, { error() {} });

    const messages = [{ role: "user", content: "x" }];
    const cases = [
        ["{", 400, "invalid JSON", "invalid_request_error"],
        [
            JSON.stringify({ messages }),
            400,
            "model must",
            "invalid_request_error",
        ],
        [
            JSON.stringify({ model: "provider-a/model-a", messages }),
            500,
            "provider unavailable",
            "provider_error",
        ],
    ];
    for (const [requestBody, status, message, type] of cases) {
        const response = await post(baseUrl, requestBody);
        assert.equal(response.status, status);
        const body = await response.json();
        assert.match(body.error.message, new RegExp(message));
        assert.equal(body.error.code, status);
        assert.equal(body.error.type, type);
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

function post(baseUrl, body, headers = {}, signal = undefined) {
    return globalThis.fetch(`${baseUrl}${CHAT_COMPLETIONS_PATH}`, {
        method: "POST",
        headers: { "content-type": "application/json", ...headers },
        signal,
        body,
    });
}

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
