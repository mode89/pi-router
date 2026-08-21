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
    RequestError,
    buildCompletion,
    buildContext,
    createChatServer,
    extractText,
    lookupReasoning,
    parseArgs,
    parseRequest,
    resolveModel,
    splitMessages,
} from "./pirouter.mjs";

const MODEL = {
    id: "model-a",
    provider: "provider-a",
    api: "test-api",
};

function fakeModels(
    modelList = [MODEL],
    complete = async () => assistantResult(),
) {
    return {
        calls: [],
        getModels(provider) {
            return provider === undefined
                ? modelList
                : modelList.filter((model) => model.provider === provider);
        },
        getModel(provider, id) {
            return modelList.find((model) => (
                model.provider === provider && model.id === id
            ));
        },
        async completeSimple(model, context, options) {
            this.calls.push({ model, context, options });
            return complete(model, context, options);
        },
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
        { role: "tool", content: "ignored" },
        { role: "assistant", content: "old answer" },
        { role: "user", content: [{ type: "text", text: "new question" }] },
    ]), {
        systemPrompt: "system\n\ndeveloper",
        conversation: [
            { role: "assistant", content: "old answer" },
            { role: "user", content: "new question" },
        ],
    });
});

test("model resolution requires exact and unambiguous matches", () => {
    const duplicate = { ...MODEL, provider: "provider-b" };
    const models = fakeModels([MODEL, duplicate, { ...MODEL, id: "other" }]);
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
    assert.equal(lookupReasoning("none"), undefined);
    assert.equal(lookupReasoning("minimal"), "minimal");
    assert.equal(lookupReasoning("max", logger), undefined);
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
            /no user\/assistant messages/,
        ],
        [
            {
                messages: [{ role: "assistant", content: "x" }],
                model: "model-a",
            },
            /last message must have role=user/,
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

const completionFormattingTest =
    "completion formatting includes reasoning, finish reason, and token usage";
test(completionFormattingTest, () => {
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

    const warnings = [];
    const failed = buildCompletion("model-a", assistantResult({
        stopReason: "error",
        errorMessage: "provider detail",
    }), { logger: { warn: (message) => warnings.push(message) } });
    assert.equal(failed.choices[0].finish_reason, "stop");
    assert.deepEqual(failed.choices[0].message, {
        role: "assistant",
        content: "answer",
    });
    assert.match(warnings[0], /provider detail/);
});

const cliParsingTest =
    "CLI parsing supports host, port, help, and rejects invalid arguments";
test(cliParsingTest, () => {
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
    assert.throws(() => parseArgs(["--sessions", "2"]), RequestError);
    assert.throws(() => parseArgs(["--port", "nope"]), /invalid port/);
});

const credentialSourceTest =
    "model runtime reads credentials from the configured auth file";
test(credentialSourceTest, async (t) => {
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

const httpSuccessTest =
    "HTTP success calls completeSimple once and returns OpenAI shape";
test(httpSuccessTest, async (t) => {
    const models = fakeModels([MODEL], async () => assistantResult({
        content: [
            { type: "thinking", thinking: "reason" },
            { type: "text", text: "result" },
        ],
    }));
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

const httpErrorsTest =
    "HTTP errors cover malformed JSON, path, model validation, "
    + "and provider failure";
test(httpErrorsTest, async (t) => {
    const duplicate = { ...MODEL, provider: "provider-b" };
    const models = fakeModels([MODEL, duplicate], async () => {
        throw new Error("provider unavailable");
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

const launcherTest =
    "launcher forwards arguments without invoking npm when dependencies "
    + "resolve";
test(launcherTest, async (t) => {
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
