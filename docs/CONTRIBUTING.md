# Contributing

## Report a bug or ask for a feature

Open an [issue](https://github.com/WASasquatch/was-node-suite-comfyui/issues). For a bug, the
ComfyUI console output and the workflow that triggered it are worth more than a description.

## Send a change

Open an issue first and say what you want to change, so a node's inputs and outputs are settled
before the code is written.

**1. Fork** the repository on GitHub.

**2. Clone your fork** into ComfyUI's `custom_nodes`, and add this repository as `upstream`:

```sh
cd ComfyUI/custom_nodes
git clone https://github.com/<your-username>/was-node-suite-comfyui.git
cd was-node-suite-comfyui
git remote add upstream https://github.com/WASasquatch/was-node-suite-comfyui.git
```

**3. Branch off `main`**, named for the change:

```sh
git switch -c image-crop-face-padding
```

**4. Make the change and run it.** Restart ComfyUI, build a graph that uses what you changed,
and test for functionality and completeness.

**5. Commit and push to your fork:**

```sh
git commit -am "Image Crop Face: pad the crop to a square"
git push -u origin image-crop-face-padding
```

**6. Open a pull request** against `main` on this repository, from your branch. Link the issue.
Where a node's inputs or outputs changed, say what a user sees when they open a workflow saved
before the change: which widgets shift, which links drop, what they retype.

Rebase on `main` before opening the pull request.

```sh
git pull --rebase upstream main
```

## The contributor documents

How a node is defined, registered and drawn is not published here. Ask in the issue and it will
be sent to you.

## What lands where

New experiments go to [WAS_Extras](https://github.com/WASasquatch/WAS_Extras) and graduate here
once they settle. The same goes for [ComfyUI_Viewer](https://github.com/WASasquatch/ComfyUI_Viewer).
