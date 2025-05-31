basic_info = """Project named: EvoX Homepage.
The ultimate goal of this project is to build a web home page for the EvoX project.
It is a new Next.js app, with only support javascript, no TypeScript support.
This project uses the latest next.js, <Link> with <a> child is invalid. Please avoid doing that.
In addition, some tools are available in this project:
- Vanilla Tailwind CSS is used for styling.
- App Router is used for routing.
- react-syntax-highlighter is used for code highlighting.
- heroicons can be used for icons.
This is a frontend only project, with no backend support.
The necessary assets are included in /public, and since it's a Next.js app, when referencing the assets, you should use the path relative to the /public folder.
For example, if you want to reference the file /public/logo.png, you should use /logo.png.
The main contents are located in /public/contents, including image resources potentially needed for the homepage.
For certain images, there are the png/jpg files and avif files, and you can use the avif files for better performance, and the png/jpg files for better compatibility.
The current file structure is as follows:
```
{structure}
```
The `layout.js` contains the base information of the page. Since it's a simple web page, you should NOT change the root layout and can only modify other configs in this file.
The `page.js` contains the main content of the page, and all contents, including the header, footer, and main content, should be placed in this file.
Individual components should be placed in the `src/components` folder.

{current_task}
"""

mutation_instruction = """

The **immediate assignment** is to make an improvement to the provided section of code that contributes to the task.
The editable section is marked with <|EDIT|> and <|END_EDIT|>.

filename: {filename}
```
{code}
```

linter output of this version: {lint}

Please provide the following in your response:

1. The **edited code**, wrapped in triple backticks (```). Limit to **200 lines**.
2. (Optional) **One new file** under the `src/components` folder if needed:

   * Include the **filename** on a single line or `None` if no file is created.
   * Followed by the **content** of the new file (if any, otherwise, leave it blank), also wrapped in triple backticks. When blank, keep the triple backticks block, but leave the content empty.
   * Only include one new file, and only if it is immediately needed (e.g., for an import in the edited code) and does not exist yet.
3. A **short commit message** describing the change.

---

Important:

* Preserve correct indentation and line breaks inside the code blocks.
* You could only add one new file, and the path should be under `src/components`.
* Do **not** include `<|EDIT|>` or `<|END_EDIT|>` tags.
* Each component should be in a separate file to keep the code clean and modular.
    * If a component file existed, but the code still uses inline code for builing the component, please delete the inline code and replace it with the component.
    * An individual component file should not include other components, only the component itself.

---

Response format, must follow strictly:

```
<edited code>
```

`<filename>`  (e.g., `src/components/NewComponent.jsx`, or `None`)
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
Placing code in the wrong place, wrong file (e.g., change the root layout in layout.js) or having very bad coding style can be considered as bad.
Do not explain, just respond with "good" or "bad".
"""
)

