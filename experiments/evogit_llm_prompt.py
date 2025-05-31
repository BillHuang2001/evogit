basic_info = """Project named: LLM auto TSP solver generation
The goal of this project is to build a program that utilizes LLM to generate good TSP solving algorithm.
To achieve this goal, the program should follows a genetic algorithm approach, which includes:
1. Generate TSP solver with LLM.
2. Evaluate the generated TSP solver with LLM.
3. Based on the evaluation, improve the TSP solver with LLM.
The current file structure is as follows:
```
{structure}
```
The main entry point of the program is `main.py`, which will import other modules and perform code generation task.

{current_task}
"""

mutation_instruction = """

The **immediate assignment** is to make a small improvement to the provided section of code that contributes to the task.
The editable section is marked with <|EDIT|> and <|END_EDIT|>.

filename: {filename}
```
{code}
```

linter output of this version: {lint}

Please provide the following in your response:

1. The **edited code**, wrapped in triple backticks (```). Limit to **200 lines**.
2. (Optional) **One new file** in the project root if needed:

   * Include the **filename** on a single line or `None` if no file is created.
   * Followed by the **content** of the new file (if any, otherwise, leave it blank), also wrapped in triple backticks. When blank, keep the triple backticks block, but leave the content empty.
   * Only include one new file, and only if it is immediately needed (e.g., for an import in the edited code) and does not exist yet.
   * It is ok to for the content in the new file to be very simple, even just a placeholder, as it will be improved in the future.
3. A **short commit message** describing the change.

---

Important:

* Preserve correct indentation and line breaks inside the code blocks.
* You could only add one new file.
* Do **not** include `<|EDIT|>` or `<|END_EDIT|>` tags.
* You only need to make an improvement to the project, so no need to make big changes, as it will be improved in the future.
* Each component should be in a separate file to keep the code clean and modular.
    * If a component file existed, but the code still uses inline code for builing the component, please delete the inline code and replace it with the component.
    * An individual component file should not include other components, only the component itself.

---

Response format, must follow strictly:

```
<edited code>
```

`<filename>`  (e.g., `foo.py` or `None`)
```
<new file content (if any)>
```

```
<short commit message>
```
"""
mutation_template = (
    basic_info
    + mutation_instruction
)

diff_template = (
    basic_info
    + """

The **immediate assignment** is judge if the following changes are good.
The changes are given in the diff format.
```
{diff}
```
Before this change, the output of the linter and the compiler is: {prev_lint}.
After this change, the output of the linter and the compiler is: {new_lint}.
If after this change, the code is relatively better, please respond with "good", otherwise, please respond with "bad".
Note that you should only consider if the changes in the diff is good or bad.
Placing code in the wrong place, wrong file or having very bad coding style can be considered as bad.
Do not explain, just respond with "good" or "bad".
"""
)


fixup_template = (
    basic_info
    + """
The immediate assignment is to make a small improvement to the provided section of code that contributes to the task.
The editable section is marked with <|EDIT|> and <|END_EDIT|>.

filename: {filename}
```
{code}
```

The purposed change is:
```
{change}
```
with the following commit message:
```
{message}
```

The change results in the following linter and compiler output: {lint}.

Please give
1. the new edited code wrapped in triple backticks. (no more than 200 lines)
2. a short commit message for this fix.
"""
)
