// Thin Claude Code adapter for the shared decision-router pruner.
//
// After a Bash call, long stdout is piped to `decision prune --json`, which
// owns every provider decision (local tiers, the JEV fallback and its switch,
// native truncation) and the ledger. This file holds no provider logic and
// never reads credentials. Any problem leaves the original result untouched.

import type { On, PluginOptions, Register } from 'claude-code';

const PRUNE_TIMEOUT_MS = 180_000;

type PruneJson = {
  text: string;
  provider: string;
  tier: number | string | null;
  original_chars: number;
  result_chars: number;
};

type BashRecord = {
  stdout?: unknown;
  stderr?: unknown;
  exitCode?: unknown;
  persistedOutputPath?: unknown;
  persistedOutputSize?: unknown;
  [key: string]: unknown;
};

export const register: Register = (on: On, options: PluginOptions) => {
  const minChars = Math.max(1000, Number(options['minChars'] ?? 12000) || 12000);

  on('tool.call', { tool: 'Bash' }, async ($, event, next) => {
    const answer = await next(event);
    try {
      if (answer.deny !== undefined) return answer;

      const prune = async (output: string): Promise<PruneJson | undefined> => {
        if (output.includes('[decision prune:')) return undefined;
        const home = (await $.env.get('HOME')) ?? '';
        const profile = (await $.env.get('CLAUDE_CONFIG_DIR')) ?? '';
        const caller = profile.endsWith('.claude-max') ? 'csmart' : 'claude';
        const run = await $.process.run(
          [`${home}/.local/bin/decision`, 'prune', '--json', '--caller', caller,
           '--command', String(event.command ?? '').slice(0, 300)],
          { stdin: output, timeoutMs: PRUNE_TIMEOUT_MS },
        );
        if (run.exitCode !== 0) return undefined;
        const pruned = JSON.parse(run.stdout) as PruneJson;
        if (typeof pruned.text !== 'string' || pruned.result_chars >= output.length) return undefined;
        return pruned;
      };
      const notify = (pruned: PruneJson, before: number) => $.ui.toast(
        `pruned ${Math.round(before / 1000)}k→${Math.round(pruned.result_chars / 1000)}k (${pruned.provider}${pruned.tier === null ? '' : ` t${pruned.tier}`})`,
        { timeoutMs: 4_000 },
      );

      if (answer.isError) {
        // A failed command: what the model reads is the error text (stdout and stderr).
        const record = answer.result as BashRecord | undefined;
        const stdout = typeof record?.stdout === 'string' ? record.stdout : '';
        const stderr = typeof record?.stderr === 'string' ? record.stderr : '';
        const errorText = typeof answer.text === 'string' ? answer.text : '';
        const exitStatus = typeof record?.exitCode === 'number'
          ? `Process exited with status ${record.exitCode}`
          : '';
        const output = [
          stdout ? `STDOUT:\n${stdout}` : '',
          stderr ? `STDERR:\n${stderr}` : '',
          !stdout && !stderr && errorText ? `ERROR OUTPUT:\n${errorText}` : '',
          exitStatus,
        ].filter(Boolean).join('\n');
        if (!output || output.length <= minChars) return answer;
        const pruned = await prune(output);
        if (!pruned) return answer;
        notify(pruned, output.length);
        return { isError: true, result: pruned.text, text: pruned.text, context: answer.context };
      }

      if (!answer.result) return answer;
      const record = answer.result as BashRecord;
      const persisted = typeof record.persistedOutputPath === 'string'
        ? record.persistedOutputPath
        : undefined;
      const savedStdout = typeof record.stdout === 'string' ? record.stdout : '';
      const stdout = persisted ? await $.fs.read(persisted) : savedStdout;
      const stderr = typeof record.stderr === 'string' ? record.stderr : '';
      const exitStatus = typeof record.exitCode === 'number'
        ? `Process exited with status ${record.exitCode}`
        : '';
      const output = [
        stdout ? `STDOUT:\n${stdout}` : '',
        stderr ? `STDERR:\n${stderr}` : '',
        exitStatus,
      ].filter(Boolean).join('\n');
      if (output.length <= minChars) return answer;
      const pruned = await prune(output);
      if (!pruned) return answer;

      const result = { ...record, stdout: pruned.text };
      delete result.persistedOutputPath;
      delete result.persistedOutputSize;
      if (persisted) result.stderr = '';
      notify(pruned, output.length);
      return { result };
    } catch {
      return answer; // pruning never breaks a Bash call
    }
  });
};
