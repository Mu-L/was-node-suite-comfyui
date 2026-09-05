"""Third-party code vendored into the pack, exempt from its source conventions.

Each subdirectory carries the upstream licence and a ``NOTICE.md`` naming the project, the commit
it was taken at, and every change made to it. Nothing here is imported at load time: the pack's
own wrappers under ``modules/model`` reach in from inside a function, so a workflow that never
uses one of these networks never pays to import it.

The pack's comment, docstring, path and custom-type conventions do not apply to this tree,
whose author cannot be asked to change it.
"""
