import js from "@eslint/js";
import stylistic from "@stylistic/eslint-plugin";

export default [
    js.configs.recommended,
    {
        plugins: {
            "@stylistic": stylistic
        },
        rules: {
            "@stylistic/eol-last": "error",
            "@stylistic/indent": ["error", 4, { SwitchCase: 1 }],
            "@stylistic/max-len": ["error", {
                code: 80,
                ignoreRegExpLiterals: true,
                ignoreTemplateLiterals: true,
                ignoreUrls: true
            }],
            "@stylistic/no-trailing-spaces": "error",
            "@stylistic/object-curly-spacing": ["error", "always"],
            "@stylistic/quotes": ["error", "double", { avoidEscape: true }],
            "@stylistic/semi": ["error", "always"]
        }
    }
];
