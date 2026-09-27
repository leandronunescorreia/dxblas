Reserved for future backend-selection helpers.

The first refactor intentionally does not introduce a custom renderer/backend
selection enum. CMake target selection will eventually be used for:

- D3D11
- D3D12
- OpenGL

without leaking one API's state model into another.
