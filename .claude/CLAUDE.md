# graphify
- **graphify** (`.claude/skills/graphify/SKILL.md`) - any input to knowledge graph. Trigger: `/graphify`
When the user types `/graphify`, invoke the Skill tool with `skill: "graphify"` before doing anything else.

# Roadmap
The authoritative plan is `.cursor/plans/ancient-chinese-search-engine_9405d973.plan.md` — **read §0 (Executable Plan v3) first**. Work the §0.6 backlog in order (Track A → E). Refer to pipeline stages by name (plan §0.3), never by number. ADRs go in `AGENTS.md` §11.

# Recommended skills for this project
- **graphify** — codebase orientation; use `graphify query "<q>"` before grepping; `graphify update .` after code changes.
- **code-review** — run on the diff before committing each §0.6 work item.
- **verify** — confirm a change actually works (run the pipeline/script, observe behavior) before marking a backlog item done.
- **security-review** — for any Track E API/upload surface (deferred), and for the corpus-poisoning / HITL-input paths (plan §14).
Not relevant: `claude-api` (this project calls Silra via the OpenAI SDK, not the Anthropic SDK).
