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

- Use existing math utilities (`mul()`, `inverse()`, `transpose()`) — never hand-roll matrix/vector operations or write helper wrappers for things the library already provides.
- Prefer to use modern C++.
- Use descriptive variable names (`mesh_id` not `mid`).
- Mark local variables `const` when they are not modified.
- Prefer `enum` / `enum class` over bare constants.
- Don't `std::move` trivially copyable types.
- Prefer `std::unordered_*` and only use others when iteration order must be deterministic.
- Keep comments minimal: no multi-line docstrings, no section separator banners. A single short line is enough when a comment is needed at all. Target an very experienced programmer.
- Very long methods can be organized with `// section` few-word comments. 

Use clang-format on the modified files.
