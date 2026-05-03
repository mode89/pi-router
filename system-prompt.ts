import type { ExtensionAPI } from "@mariozechner/pi-coding-agent";

export default function (pi: ExtensionAPI) {
    pi.on("before_agent_start", async (event, ctx) => {
        const stripped = event.systemPrompt
            .replace(/^Current date: .*\n?/m, "")
            .replace(/^Current working directory: .*\n?/m, "");

        return {
            systemPrompt: stripped,
        };
    });
}
