# Prompt files

One prompt per line. Use them to try the chat model or to run a batch test.

| File | Lines | What it is |
|------|-------|------------|
| `famous_lines.txt` | 50 | Well-known lines from Shakespeare's plays and sonnets |
| `questions.txt` | 150 | Old-style questions ("Dost thou love me truly?") |
| `test_lines.txt` | 227 | Old-style greetings, oaths, commands, statements, famous-style lines and short fragments |
| `modern_lines.txt` | 40 | Modern English ("What is the weather today?"), included to show where the model fails |

## Run a whole file

```bash
./.venv/bin/python run_prompts.py prompts/test_lines.txt --limit 20
./.venv/bin/python run_prompts.py prompts/questions.txt --out replies_questions.txt
./.venv/bin/python run_prompts.py prompts/modern_lines.txt --temperature 0.8
```

On Windows use `.venv\Scripts\python.exe run_prompts.py ...`.

## What to expect

The model is a character-level Shakespeare imitator, not an assistant. Replies keep the play format
(`NAME:` then lines) and mostly use real words at temperature 0.5, but they usually do **not** answer the
prompt. Some replies are better than others; the files are for exploring that, not a benchmark with right answers.
Some famous lines may also appear in the training text, so a good continuation can be memorised rather than invented.
