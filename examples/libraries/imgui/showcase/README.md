# ImGui workbench

A live SDL and Dear ImGui application demonstrating SDL event translation,
ImGui input forwarding, persistent widget values, text entry, and mouse and
keyboard capture reporting.

Run from the repository root:

```sh
cog run examples/libraries/imgui/showcase
```

The app project is `0.1.0` and uses `cog-imgui` `0.1.0` plus `cog-sdl`
`0.13.0`. Cog requires the SDL package alias directly so the source module
`sdl_lib` is visible to this project; ImGui uses the same SDL package as its
backend.

Toggle diagnostics, adjust the preview slider, edit the workspace field, and
press **Rebuild preview** to see the widget state update across frames.
