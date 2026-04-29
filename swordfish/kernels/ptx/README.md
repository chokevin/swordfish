# Raw PTX vector-add spike

This is the smallest raw-PTX learning artifact in the repo: a handwritten
float32 vector-add kernel string plus an explicit Python loader blocker.

Current blocker: the repo does not yet include a CUDA driver loader for raw PTX.
The next implementation step is to use CUDA driver bindings on a CUDA Linux host
to load `PTX_VECTOR_ADD_F32`, launch `vector_add_f32`, and compare against
`torch_vector_add_reference`.

Until that loader exists, `ptx_vector_add(...)` fails loudly instead of falling
back to `torch.add`.
