# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Preston Logan. Part of qwen3.8-flash-next-fp4-local-spark.
# Worker extension: run a script inside the vLLM worker process (sees `worker`, sets RESULT).
class Ext:
    def fx_exec(self, path: str) -> str:
        g = {"worker": self, "__name__": "fx_exec"}
        with open(path) as f:
            exec(compile(f.read(), path, "exec"), g)
        return str(g.get("RESULT", "ok"))
