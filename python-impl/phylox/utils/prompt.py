"""This module contruct high-level prompt for a code base."""

from pathlib import Path
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

    def visit(node):
        if node.type == "function_declaration":
            sig = extract_function_signature(node, code.encode("utf8"))
            if sig:
                symbols.append(sig)
        elif node.type == "class_declaration":
            sig = extract_class_signature(node, code.encode("utf8"))
            if sig:
                symbols.append(sig)
        for child in node.children:
            visit(child)

    visit(root)
    return symbols


def project_tree_view(project_path: str, mode: str = "json"):
    """Get a tree view of the project structure."""
    if mode == "json":
        args = ["tree", "-Ji"]
    elif mode == "path":
        args = ["tree", "-fi"]
    elif mode == "tree":
        args = ["tree"]
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    tree_view = subprocess.run(
        args + [project_path],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    return tree_view


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
