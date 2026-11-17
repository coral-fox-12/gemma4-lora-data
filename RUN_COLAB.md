# Run Gemma 4 LoRA on free Colab T4

**What this is:** QLoRA training of `google/gemma-4-E4B-it` on `train_mix_v2.jsonl`
(3,738 examples, ~6.3M tokens) — the file the Phase 2 work built. The base model
scores 0/10 on the dev slice; this LoRA is the thing that has to lift it.
**Time:** ~2–2.5 h per epoch on a T4 (estimate), 2 epochs, checkpoints saved
after each epoch — if Colab disconnects mid-run, the epoch-1 checkpoint is
already saved and usable.

**One account only.** Use a single Google account for this Colab session. Do
NOT run the same training in parallel on several accounts — that is the
highest-ban-risk use of Colab and can get the accounts suspended.

## Steps (phone-friendly)

### 1. Open the notebook — one tap
Open this link in Chrome on your phone (or desktop):

**https://colab.research.google.com/github/coral-fox-12/gemma4-lora-data/blob/main/colab_lora.ipynb**

If that link fails, go to https://colab.research.google.com → **File → Open
notebook → GitHub** → paste `coral-fox-12/gemma4-lora-data` → pick
`colab_lora.ipynb`.

### 2. Turn on the T4 GPU
Top menu: **Runtime → Change runtime type → Hardware accelerator → T4 GPU →
Save.** (If T4 is not offered, any GPU listed there works — the notebook
checks and prints what it got. If it says "No GPU", stop: free-tier GPU is
not guaranteed at that moment, try again later.)

### 3. Run all
**Runtime → Run all.** The notebook then does everything itself, in order:
1. checks the GPU + installs packages (transformers, peft, bitsandbytes…)
   — if it asks you to restart the runtime after install, tap
   **Runtime → Restart session** and run all again (installs persist);
2. downloads the training data from GitHub and verifies its sha256
   (no upload needed — but if the download ever fails, the notebook
   falls back to a file picker: upload `train_mix_v2.jsonl`);
3. loads the model in 4-bit (NF4) with `attn_implementation="sdpa"`
   (the only attention path that works for Gemma 4 — see NOTES.md);
4. trains 2 epochs, saving a checkpoint after each epoch;
5. zips the final adapter and triggers a download
   (`gemma4-e4b-lora.zip`, ~70 MB).

Just leave the tab open while it trains. Don't close the tab or let the
phone sleep with the browser killed — a disconnect stops the run
(the epoch checkpoint still survives if at least one epoch finished).

### 4. Get the adapter
- **Phone download:** when the run finishes, the browser downloads
  `gemma4-e4b-lora.zip` (~70 MB). Send that file to the agent afterwards.
- **Drive (recommended):** run the last cell ("OPTIONAL — save to Drive") and
  the adapter is copied to `MyDrive/gemma4-e4b-lora` — no download needed,
  the agent picks it up from there.

## If Colab disconnects
- **During epoch 1:** nothing saved yet. Re-open the notebook and Run all
  again (a fresh session).
- **After epoch 1:** the checkpoint `gemma4-e4b-lora/checkpoint-234` is in
  the session — but it only survives if you download it or copy it to Drive
  BEFORE the session dies. If the session is already gone, just re-run:
  epoch 1 takes ~2 h, and the run is cheap to repeat.
- **Resume:** the notebook has a `RESUME_FROM` setting at the top of the
  training cell — set it to a checkpoint path (e.g.
  `/content/gemma4-e4b-lora/checkpoint-234`) and re-run from the training
  cell to continue instead of starting over. (Only works if the checkpoint
  files are still on disk / in Drive — copy them back to `/content/` first.)

## What success looks like
- `loss` in the training log falls from ~2.5–3.5 at step 1 to ~0.5–1.5 by
  the end of epoch 1. (Exact numbers vary — a flat or rising loss means
  something is wrong; stop and tell the agent.)
- Final cell prints `saved: /content/gemma4-e4b-lora/final` and the zip
  downloads.
- The adapter is then evaluated on Kaggle L4x4 (the agent does that part)
  — target: clearly above the 0/10 base. Kill criteria: scaffold v1 < 0.20
  by ~24/11, or no LoRA gain by ~27/11 → stop.

## If the T4 route fails
Fall back to the Kaggle GPU refresh (21/11, L4x4, 3.78 h left) — same data,
same config, the notebook's training cell ports 1:1 to a Kaggle notebook.
Tell the agent which path failed and it will set that up.
