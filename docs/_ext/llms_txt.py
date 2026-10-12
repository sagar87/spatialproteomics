"""Sphinx extension that publishes the documentation in a form that LLMs and coding agents can read.

After an HTML build, this writes into the output directory:

- ``<page>.md`` for every documentation page: notebooks as Markdown with code and short text outputs
  (no images), API pages rendered from the docstrings, and other pages converted from reStructuredText.
- ``llms.txt``: a curated index following https://llmstxt.org, linking to the Markdown pages.
- ``llms-full.txt``: all Markdown pages concatenated into one file.

Configuration (in ``conf.py``):

- ``llms_txt_base_url``: absolute URL of the published documentation, used for links in ``llms.txt``.
- ``llms_txt_summary``: one-paragraph project summary (the blockquote in ``llms.txt``).
- ``llms_txt_details``: Markdown placed between the summary and the link sections.
- ``llms_txt_descriptions``: ``{docname: description}`` shown next to each link.
"""

import importlib
import inspect
import re
import shutil
import subprocess
from pathlib import Path

import nbformat
import xarray as xr
from sphinx.util import logging

logger = logging.getLogger(__name__)

MAX_OUTPUT_LINES = 15
MAX_DEFAULT_REPR = 60


# --- notebooks ---------------------------------------------------------------


def _clean_markdown_cell(text: str) -> str:
    # replace HTML images with their alt text and drop the wrapping paragraph tags
    text = re.sub(r"<img[^>]*?alt=\"([^\"]*)\"[^>]*>", lambda m: f"*[Figure: {m.group(1)}]*", text, flags=re.IGNORECASE)
    text = re.sub(r"<img[^>]*>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"</?p[^>]*>", "", text, flags=re.IGNORECASE)
    # drop inline (base64) images, keep links to image files
    text = re.sub(r"!\[([^\]]*)\]\(data:[^)]*\)", "", text)
    text = re.sub(r"!\[([^\]]*)\]\(attachment:[^)]*\)", "", text)
    return text.strip()


def _text_outputs(cell) -> str:
    chunks = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream" and output.get("name") == "stdout":
            chunks.append(output.get("text", ""))
        elif output.get("output_type") in ("execute_result", "display_data"):
            data = output.get("data", {})
            # keep the plain text version of results (e.g. dataset and data frame reprs), but skip images
            if "text/plain" in data and not any(k.startswith("image/") for k in data):
                text = data["text/plain"]
                if not re.match(r"^<(Figure|matplotlib|Axes)", text):
                    chunks.append(text)
    lines = "".join(chunks).strip("\n").splitlines()
    if len(lines) > MAX_OUTPUT_LINES:
        lines = lines[:MAX_OUTPUT_LINES] + ["..."]
    return "\n".join(lines)


def notebook_to_markdown(path: Path) -> str:
    nb = nbformat.read(str(path), as_version=4)
    parts = []
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            text = _clean_markdown_cell(cell.source)
            if text:
                parts.append(text)
        elif cell.cell_type == "code":
            # IPython magics are irrelevant outside of jupyter
            code = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith("%")).strip()
            if not code:
                continue
            parts.append(f"```python\n{code}\n```")
            output = _text_outputs(cell)
            if output:
                parts.append(f"Output:\n\n```text\n{output}\n```")
    return "\n\n".join(parts) + "\n"


# --- API pages ---------------------------------------------------------------


def _accessor_names() -> dict:
    """Maps accessor classes to the attribute name they are registered under on xr.Dataset."""
    names = {}
    for name, value in vars(xr.Dataset).items():
        accessor = getattr(value, "_accessor", None)
        if accessor is not None:
            names[accessor] = name
    return names


TYPE_ALIASES = {
    "xarray.core.dataset.Dataset": "xr.Dataset",
    "numpy.ndarray": "np.ndarray",
    "pandas.core.frame.DataFrame": "pd.DataFrame",
    "pandas.DataFrame": "pd.DataFrame",
}


def _shorten_types(text: str) -> str:
    for long, short in TYPE_ALIASES.items():
        text = text.replace(long, short)
    return text


def _format_signature(func, drop_self: bool) -> str:
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return "(...)"
    params = list(sig.parameters.values())
    if drop_self and params and params[0].name == "self":
        params = params[1:]
    formatted = []
    for p in params:
        if p.default is not inspect.Parameter.empty:
            if callable(p.default) and getattr(p.default, "__name__", "") == "<lambda>":
                default = "<lambda>"
            else:
                default = repr(p.default)
                if len(default) > MAX_DEFAULT_REPR:
                    closing = default[-1] if default[-1] in "]})" else ""
                    default = f"{default[: MAX_DEFAULT_REPR - 5]} ...{closing}"
            p = p.replace(default=inspect.Parameter.empty)
            formatted.append(f"{p}={default}")
        else:
            formatted.append(str(p))
    text = f"({', '.join(formatted)})"
    if sig.return_annotation is not inspect.Signature.empty:
        text += f" -> {inspect.formatannotation(sig.return_annotation)}"
    return _shorten_types(text)


def _format_docstring(obj) -> str:
    doc = inspect.getdoc(obj) or ""
    # numpydoc section underlines would turn into Markdown headings, so turn the section titles into bold text
    doc = re.sub(r"^([A-Z][A-Za-z ]+)\n-{3,}$", r"**\1**\n", doc, flags=re.MULTILINE)
    return doc.strip()


def _document(qualname: str, obj, drop_self: bool) -> str:
    entry = f"### `{qualname}{_format_signature(obj, drop_self)}`"
    doc = _format_docstring(obj)
    return f"{entry}\n\n{doc}" if doc else entry


def module_to_markdown(module_name: str) -> str:
    module = importlib.import_module(module_name)
    accessors = _accessor_names()
    # the public name of the module's subpackage, e.g. sp.pp for spatialproteomics.pp.preprocessing
    package = module_name.rsplit(".", 1)[0].replace("spatialproteomics", "sp", 1)

    sections = []
    functions = [
        (name, obj)
        for name, obj in vars(module).items()
        if inspect.isfunction(obj) and obj.__module__ == module_name and not name.startswith("_")
    ]
    classes = [
        (name, obj)
        for name, obj in vars(module).items()
        if inspect.isclass(obj) and obj.__module__ == module_name and not name.startswith("_")
    ]

    for cls_name, cls in classes:
        accessor = accessors.get(cls)
        prefix = f"ds.{accessor}" if accessor else f"sp.{cls_name}"
        header = f"## `{prefix}`" + (f" ({cls_name})" if accessor else "")
        lines = [header]
        if cls.__doc__:
            lines.append(_format_docstring(cls))
        if accessor:
            lines.append(
                f"Available on any `xarray.Dataset` after `import spatialproteomics as sp`, as `ds.{accessor}`."
            )
        if accessor and "__getitem__" in vars(cls) and vars(cls)["__getitem__"].__doc__:
            lines.append(f"### `ds.{accessor}[...]`\n\n{_format_docstring(vars(cls)['__getitem__'])}")
        for name, member in vars(cls).items():
            if name.startswith("_") or not inspect.isfunction(member):
                continue
            lines.append(_document(f"{prefix}.{name}", member, drop_self=True))
        sections.append("\n\n".join(lines))

    if functions:
        lines = [
            f"## Functions for the spatialdata backend (`{package}`)",
            "These functions take a `spatialdata.SpatialData` object as first argument "
            "and modify it in place unless `copy=True`.",
        ]
        for name, func in functions:
            lines.append(_document(f"{package}.{name}", func, drop_self=False))
        sections.append("\n\n".join(lines))

    return "\n\n".join(sections) + "\n"


# --- reStructuredText pages --------------------------------------------------


def rst_to_markdown(path: Path) -> str:
    text = path.read_text()
    # sphinx directives that pandoc does not understand; automodule is handled separately
    text = re.sub(r"^\.\. (highlight|toctree)::.*?(?=^\S|\Z)", "", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r"^\.\. automodule::.*?(?=^\S|\Z)", "", text, flags=re.MULTILINE | re.DOTALL)
    text = re.sub(r":code:`([^`]*)`", r"``\1``", text)
    if shutil.which("pandoc"):
        result = subprocess.run(
            ["pandoc", "--from=rst", "--to=gfm", "--wrap=none"], input=text, capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout
        logger.warning(f"llms_txt: pandoc failed for {path}, using raw reStructuredText: {result.stderr}")
    return text


# --- assembling the output ---------------------------------------------------


def _ordered_docnames(env, root: str) -> list:
    """Docnames in the order in which they appear in the toctrees, starting at the root document."""
    ordered = []

    def visit(docname):
        if docname in ordered:
            return
        ordered.append(docname)
        for child in env.toctree_includes.get(docname, []):
            visit(child)

    visit(root)
    return ordered


def _toctree_captions(srcdir: Path, root: str) -> list:
    """Returns [(caption, [docnames])] for each toctree in the root document."""
    text = (srcdir / f"{root}.rst").read_text()
    sections = []
    for block in re.findall(r"^\.\. toctree::\n((?:[ \t]+.*\n|[ \t]*\n)+)", text, flags=re.MULTILINE):
        caption = re.search(r":caption:\s*(.+)", block)
        entries = [line.strip() for line in block.splitlines() if line.strip() and not line.strip().startswith(":")]
        sections.append(((caption.group(1).strip().rstrip(":") if caption else "Docs"), entries))
    return sections


def _page_markdown(app, docname: str) -> str:
    source = Path(app.env.doc2path(docname))
    title = app.env.titles[docname].astext() if docname in app.env.titles else docname

    if source.suffix == ".ipynb":
        return notebook_to_markdown(source)

    markdown = rst_to_markdown(source)
    modules = re.findall(r"^\.\. automodule::\s*(\S+)", source.read_text(), flags=re.MULTILINE)
    for module_name in modules:
        markdown = markdown.rstrip() + "\n\n" + module_to_markdown(module_name)
    if not markdown.lstrip().startswith("#"):
        markdown = f"# {title}\n\n{markdown}"
    return markdown


def build_finished(app, exception):
    if exception is not None or app.builder.format != "html":
        return

    config = app.config
    outdir = Path(app.outdir)
    root = config.root_doc
    base_url = config.llms_txt_base_url.rstrip("/") + "/"
    pages = {}

    for docname in _ordered_docnames(app.env, root):
        try:
            markdown = _page_markdown(app, docname)
        except Exception as e:  # never fail the documentation build because of this
            logger.warning(f"llms_txt: could not convert {docname}: {e}")
            continue
        target = outdir / f"{docname}.md"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(markdown)
        pages[docname] = markdown

    # llms.txt
    lines = [f"# {config.project}", "", f"> {config.llms_txt_summary.strip()}", ""]
    if config.llms_txt_details:
        lines += [config.llms_txt_details.strip(), ""]
    for caption, docnames in _toctree_captions(Path(app.srcdir), root):
        lines.append(f"## {caption}")
        lines.append("")
        for docname in docnames:
            if docname not in pages:
                continue
            title = app.env.titles[docname].astext()
            description = config.llms_txt_descriptions.get(docname)
            link = f"- [{title}]({base_url}{docname}.md)"
            lines.append(f"{link}: {description}" if description else link)
        lines.append("")
    lines += [
        "## Optional",
        "",
        f"- [Full documentation]({base_url}llms-full.txt): all pages above concatenated into a single file",
        "- [Source code](https://github.com/sagar87/spatialproteomics): GitHub repository",
        "",
    ]
    (outdir / "llms.txt").write_text("\n".join(lines))

    # llms-full.txt
    full = [f"# {config.project}\n\n> {config.llms_txt_summary.strip()}\n"]
    for docname, markdown in pages.items():
        if docname == root:
            continue
        full.append(f"<!-- source: {base_url}{docname}.html -->\n\n{markdown.strip()}\n")
    (outdir / "llms-full.txt").write_text("\n\n---\n\n".join(full))

    logger.info(f"llms_txt: wrote llms.txt, llms-full.txt and {len(pages)} Markdown pages")


def setup(app):
    app.add_config_value("llms_txt_base_url", "", "env")
    app.add_config_value("llms_txt_summary", "", "env")
    app.add_config_value("llms_txt_details", "", "env")
    app.add_config_value("llms_txt_descriptions", {}, "env")
    app.connect("build-finished", build_finished)
    return {"version": "0.1", "parallel_read_safe": True, "parallel_write_safe": True}
