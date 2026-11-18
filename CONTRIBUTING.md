# Contributing

Thank you for looking at this project. Questions, bug reports and pull requests are welcome.

## Set up

```bash
git clone https://github.com/WilliamQiuzy/echo-view-routing
cd echo-view-routing
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest          # about two minutes on a laptop; needs no GPU, no weights and no dataset
ruff check src tests tools
```

Three groups of tests are left out by default because they need something that is not in the repository:

| Marker | Needs | Run with |
|---|---|---|
| `weights` | the published weights (`hf auth login` or `ECHOVIEW_WEIGHTS`) | `pytest -m weights` |
| `gpu` | a CUDA GPU | `pytest -m gpu` |
| `ev9v` | the EV9V dataset (`ECHOVIEW_EV9V`) | `pytest -m ev9v` |

## What a change needs

- **A test.** Write the test first and watch it fail. A bug fix comes with a test that fails without the fix.
- **Passing checks.** `pytest` and `ruff check` must pass. CI runs both on every pull request.
- **Unchanged numbers.** The windows, the decoder, the metrics and the decoder rule are pinned to the outputs of the code that produced the published numbers (`tests/test_windows.py`, `test_decoder.py`, `test_evaluation_parity.py`). A change that moves any of them is a change of the method. Say so in the pull request and explain why.
- **Plain words.** Comments and documentation use short full sentences and define their terms.

## Data rules

These rules are not negotiable, because they come from the data's licences.

- **No patient data.** Never put a recording, a frame, an identifier or a per-recording output of a restricted dataset (MIMIC-IV-Echo, EchoNet) in an issue, a pull request, a test or a commit. The tests of the readers use synthetic files only.
- **No recording names.** Definition files name EV9V recordings by key, not by name.
- **No weights in git.** Weights live on the Hugging Face Hub.
- **MIMIC labels are not a reference standard.** A number measured against another classifier's labels is agreement, not accuracy. Write it that way.

## Reporting a problem

Open an issue with the command you ran, what you expected and what happened. Add the output of `echoview weights` and your Python and PyTorch versions. For a file that cannot be read, describe its format (vendor, compression, bit depth) instead of attaching it.
