# External library examples

Each project here declares its own version as `0.1.0` in `package.kdl`. Its
`dependencies.kdl` pins its library dependency to the latest available
implementation:

| Library | Showcase | Dependency version |
| --- | --- | --- |
| raylib | [field lab](raylib/showcase/) | `cog-raylib 0.1.18` |
| SDL | [event canvas](sdl/showcase/) | `cog-sdl 0.13.0` |
| cogweb | [API showcase](cogweb/showcase/) | `cogweb 0.3.0` |
| ImGui | [workbench](imgui/showcase/) | `cog-imgui 0.1.0` (uses its `cog-sdl 0.13.0` dependency) |

The workbench imports a small local `sdl_lib` source module for typed event
codes. The native `sdl` module comes from ImGui's `cog-sdl` dependency, avoiding
a second registration of the same SDL plugin. These four showcases were
written for this repository.

Run a project from the repository root after installing its listed package in
the selected Cog store. For example:

```sh
cog run examples/libraries/raylib/showcase
cog run examples/libraries/sdl/showcase
cog run examples/libraries/imgui/showcase
cog run examples/libraries/cogweb/showcase
```
