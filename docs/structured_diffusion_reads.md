# Structured diffusion reads

The `/v1/diffusion/reads` endpoint runs one read-only model pass over a seeded
DiffusionGemma canvas. It returns temperature-1 log probabilities for the token
IDs you ask for. It does not write the canvas to the KV cache.

The server loads the requested model if needed. A shared cache lock keeps that
exact model loaded until the read ends.

Start the server with a DiffusionGemma checkpoint:

```bash
mlx_vlm.server \
  --model mlx-community/diffusiongemma-26B-A4B-it-4bit \
  --port 8080
```

Send prompt tokens, an active seed canvas, and one or more read slots. This
example builds a three-token canvas:

```python
seed_canvas = [236743, 42, 107]
request = {
    "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
    "input_ids": [2, 3, 4],
    "seed_canvas": seed_canvas,
    "slots": [{"position": 1, "token_ids": [562, 603]}],
    "candidate_only": True,
}
```

`seed_canvas` may use any active length up to the model's canvas limit. Every
token ID must be in the model vocabulary. Each slot must be unique and inside
the active canvas.

Set `candidate_only` when the requested IDs form a closed answer set. It skips
the full vocabulary output head. Its scores and diagnostics cover only the
requested IDs.

The response keeps token IDs and log probabilities in request order:

```json
{
  "model": "mlx-community/diffusiongemma-26B-A4B-it-4bit",
  "reads": [
    {
      "position": 1,
      "token_id": 562,
      "token_logprob": -0.139,
      "entropy": 0.386,
      "token_ids": [562, 603],
      "logprobs": [-0.139, -2.039]
    }
  ],
  "usage": {
    "prompt_tokens": 3,
    "denoising_steps": 1,
    "candidate_only": true
  }
}
```

The usage block confirms the scoring mode used by the server.

Without `candidate_only`, `token_id`, `token_logprob`, and `entropy` cover the
full vocabulary. `logprobs` holds the score for each requested token. With
`candidate_only`, every returned value covers only the requested tokens.

The server groups matching `candidate_only` requests for up to 2 ms. Each
request keeps batch size one, so it uses the same 4-bit kernels and scores as a
single request. MLX then evaluates the queued graphs together. A group holds at
most four requests by default.

Set `MLX_VLM_DIFFUSION_READ_BATCH_COALESCE_MS` to change the wait. Set
`MLX_VLM_DIFFUSION_READ_MAX_BATCH_SIZE` to change the group limit. Use `0` for
the wait to favor single-request latency.
