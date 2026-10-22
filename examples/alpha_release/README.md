# Experimental candidate examples

No checkpoint currently has an approved alpha contract. These examples exercise the retained native interfaces for engineering evaluation. They do not advertise certification.

Use a local copy of a pinned checkpoint from the release support matrix:

```sh
python examples/alpha_release/native.py /path/to/checkpoint
mlx-one generate /path/to/checkpoint "The city was" --offline --max-tokens 16
mlx-one serve /path/to/checkpoint --offline --inference-mode reference_dense
python examples/alpha_release/http.py
python examples/alpha_release/http.py --stream
python examples/alpha_release/http.py --chat --stream
```

The server starts on port 8888. Chat requires a checkpoint with an applicable chat template; GPT-2 only supports completion in this matrix. Set `MLX_ONE_API_KEY` in both server and example environments to use the documented Bearer authentication.
