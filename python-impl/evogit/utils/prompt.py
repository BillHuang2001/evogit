"""This module contruct high-level prompt for a code base."""

# %%
from pathlib import Path
import json
import os
import random
import subprocess
from tree_sitter import Language, Parser
import tree_sitter_javascript as tsjavascript
import tree_sitter_html as tshtml
import tree_sitter_css as tscss


def get_source_files(root_dir, extensions=(".js", ".jsx", ".html")):
    return list(Path(root_dir).rglob("*"))  # optionally filter by suffix


def extract_function_signature(node, code):
    name_node = node.child_by_field_name("name")
    params_node = node.child_by_field_name("parameters")
    if not name_node or not params_node:
        return None

    name = code[name_node.start_byte : name_node.end_byte].decode("utf8")
    params = code[params_node.start_byte : params_node.end_byte].decode("utf8")
    return f"function {name}{params}"


def extract_class_signature(node, code):
    name_node = node.child_by_field_name("name")
    super_node = node.child_by_field_name("superclass")

    name = code[name_node.start_byte : name_node.end_byte].decode("utf8")
    if super_node:
        superclass = code[super_node.start_byte : super_node.end_byte].decode("utf8")
        return f"class {name} extends {superclass}"
    else:
        return f"class {name}"


def extract_signatures(code, language):
    parser = Parser(language)
    tree = parser.parse(bytes(code, "utf8"))
    root = tree.root_node

    symbols = []

    def visit(node, parent_hierarchy=None):
        if parent_hierarchy is None:
            parent_hierarchy = []

        current_hierarchy = parent_hierarchy[:]
        if node.type == "function_declaration":
            sig = extract_function_signature(node, code.encode("utf8"))
            if sig:
                current_hierarchy.append(sig)
                symbols.append(
                    {
                        "type": "function",
                        "signature": sig,
                        "hierarchy": current_hierarchy,
                    }
                )
        elif node.type == "class_declaration":
            sig = extract_class_signature(node, code.encode("utf8"))
            if sig:
                current_hierarchy.append(sig)
                symbols.append(
                    {"type": "class", "signature": sig, "hierarchy": current_hierarchy}
                )
        elif node.type == "variable_declaration":
            for child in node.children:
                if child.type == "variable_declarator":
                    name_node = child.child_by_field_name("name")
                    if name_node:
                        name = code[name_node.start_byte : name_node.end_byte].decode(
                            "utf8"
                        )
                        current_hierarchy.append(name)
                        symbols.append(
                            {
                                "type": "variable",
                                "name": name,
                                "hierarchy": current_hierarchy,
                            }
                        )

        for child in node.children:
            visit(child, current_hierarchy)

    visit(root)
    return symbols


def project_tree_view(project_path: str, mode: str = "json"):
    """Get a tree view of the project structure."""
    if mode == "json":
        args = ["tree", "-Ji", "--gitignore"]
    elif mode == "path":
        args = ["tree", "-fi", "--gitignore"]
    elif mode == "tree":
        args = ["tree", "--gitignore"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    tree_view = subprocess.run(
        args + [project_path],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return tree_view


def _flatten_tree_view(tree_view: str):
    file_list = []
    for file in tree_view:
        if file["type"] == "file":
            file_list.append(file)
        elif file["type"] == "directory":
            dir_name = file["name"]
            sub_files = _flatten_tree_view(file["contents"])
            sub_files = [
                {**sub_file, "name": os.path.join(dir_name, sub_file["name"])}
                for sub_file in sub_files
            ]
            file_list.extend(sub_files)
        else:
            raise ValueError(f"Unsupported file type: {file['type']}")


def random_sample_file(project_path: str, tree_view: str):
    """Randomly sample a file from the project."""
    tree_view = json.loads(tree_view)
    flattened_list = _flatten_tree_view(tree_view)
    selected_file = random.choice(flattened_list)
    return os.path.join(project_path, selected_file["name"])


def get_language_by_extension(file_extension):
    """Get the appropriate language parser based on file extension."""
    if file_extension in {".js", ".jsx"}:
        return Language(tsjavascript.language())
    elif file_extension == ".html":
        return Language(tshtml.language())
    elif file_extension == ".css":
        return Language(tscss.language())
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")


def get_linter_feedback(project_path: str, project_type: str, async_run: bool = False):
    if project_type == "python":
        return get_python_feedback(project_path, async_run)
    elif project_type == "nextjs":
        return get_nextjs_feedback(project_path, async_run)


def get_nextjs_feedback(project_path: str, async_run: bool = False):
    """Get ESLint feedback for the project.
    If async_run is True, run ESLint in the background and return the handler.
    Otherwise, run ESLint synchronously and return the output as string.
    """
    if async_run:
        handler = subprocess.Popen(
            "npm install --silent && npx next lint -f codeframe && npx next build --no-lint --no-mangling",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_path,
            shell=True,
        )
        return handler
    else:
        lint_output = subprocess.run(
            "npm install --silent && npx next lint -f codeframe && npx next build --no-lint --no-mangling",
            check=True,
            text=True,
            capture_output=True,
            cwd=project_path,
            shell=True,
        ).stdout
        return lint_output


def get_python_feedback(project_path: str, async_run: bool = False):
    """Run Ruff to get lint feedback for the python project.
    If async_run is True, run Ruff in the background and return the handler.
    Otherwise, run Ruff synchronously and return the output as string.
    """
    # Disable the current python virtual environment
    clean_env = os.environ.copy()
    clean_env.pop("VIRTUAL_ENV", None)
    clean_env["UV_CACHE_DIR"] = "/tmp/uv_cache"
    # sed "s|$(pwd)/||" is used to replace the absolute path with a relative one
    if async_run:
        handler = subprocess.Popen(
            'uv run pyright . | sed "s|$(pwd)/||"',
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=project_path,
            shell=True,
            env=clean_env,
        )
        return handler
    else:
        lint_output = subprocess.run(
            'uv run pyright . | sed "s|$(pwd)/||"',
            check=True,
            text=True,
            capture_output=True,
            cwd=project_path,
            shell=True,
            env=clean_env,
        ).stdout
        return lint_output
