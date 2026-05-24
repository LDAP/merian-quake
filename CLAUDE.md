# Project

Vulkan C++ project. Main work happens in `subprojects/merian`.

Use the same bash commands to prevent re-approval.

# Build & Run

Uses Meson.

Setup: Run once if `build` does not exist `meson setup build`
Compile: `meson compile -C build`
Run `build/merian-quake`, better with timeout: `timeout 15 build/merian-quake`

# Tests

Enable tests: `meson configure build -Dmerian:tests=true`
Run one: `build/subprojects/merian/tests/test-<name>` (e.g. `test-small-vector`)

# Coding style

 Comments
  - Single short line is the default; multi-line walls of text are out. If the explanation needs a
  paragraph, the code probably needs restructuring instead.
  - Explain why, never what. Identifier names already say what. Don't mention implementation details users of a function, class or interface!
  - Inside long methods, label sub-sections with one-liner comments (// 1. ..., // upload prev vertices)
  — never banner separators.
  - File-level major dividers (// --- Section ---) are allowed sparingly for the obvious lifecycle splits
   (constructor / building / update). Don't multiply them.
  - Drop comments that just point to commit history, removed files, or the old pipeline (e.g. "matches 
  the legacy motion-vector computation in quake_helpers.cpp").
  - Keep TODOs as commented-out future code when the code is the clearest spec for the deferred work;
  otherwise delete.
  - Do NOT comment about how subclasses might override, design alternatives considered, or future intent.
   That belongs in the PR, not the code.


  Naming
  - Descriptive: mesh_id, node_id, vertex_count, prim_count — not mid, nid, vc.
  - Tight-scope math locals can be terse (m, it, v, pv) but only when the surrounding code makes the role
   obvious.
  - Method names follow lifecycle verb conventions: add_*, mark_*_dirty, upload_*, compute_*, build_*,
  ensure_*. One verb per concept; pick one and stick to it.

  Code
  - const on every local that isn't reassigned (the reference does this religiously).
  - Modern containers / idioms: try_emplace, extract, structured bindings, auto [it, inserted], assign(n,
   value), std::move on heavy types only.
  - Replace hand-rolled matrix building with the library: mul, transpose, inverse, identity, translation,
   scale, rotation. Never write a 3-line "AngleVectors then fix-up" snippet when it's needed twice.
  - Use enum / enum class over magic constants.
  - Prefer std::unordered_map over std::map unless iteration order matters.

Use clang-format on the modified files.
