import { readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { join } from "node:path";

const decision = join(homedir(), ".local", "bin", "decision");
const config = join(homedir(), ".config", "decision-router", "config.json");

function threshold(): number {
  try {
    const value = JSON.parse(readFileSync(config, "utf8"))?.prune?.min_output_chars;
    return Number.isInteger(value) && value >= 0 ? value : 16000;
  } catch {
    return 16000;
  }
}

const plugin = {
  name: "decision-prune",
  manifest: { capabilities: ["hooks"] },
  hooks: {
    afterTool(context: any) {
      if (context?.tool?.name !== "run_commands") return;
      const original = context?.result?.output;
      if (!Array.isArray(original)) return;
      const minimum = threshold();
      let changed = false;
      const output = original.map((item: any) => {
        if (!item || typeof item.result !== "string") return item;
        if (item.result.length <= minimum || item.result.includes("[decision prune:")) {
          return item;
        }
        try {
          const run = spawnSync(decision, ["prune", "--caller", "cline", "--command", String(item.query ?? "").slice(0, 2000)], {
            input: item.result,
            encoding: "utf8",
            timeout: 2500,
            maxBuffer: 16 << 20,
            env: { ...process.env, DECISION_ROUTER_CALLER: "cline" },
          });
          if (run.status !== 0 || run.error || !run.stdout) return item;
          changed = true;
          return { ...item, result: run.stdout };
        } catch {
          return item;
        }
      });
      if (changed) {
        return { result: { ...context.result, output } };
      }
    },
  },
};

export default plugin;
