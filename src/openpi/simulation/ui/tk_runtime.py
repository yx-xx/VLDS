"""Use the desktop's Tk font renderer when a standalone Python ships a bitmap-only Tk."""

import importlib
import importlib.util
import pathlib
import sys
import sysconfig


def _desktop_tk():
    # System extensions with the same CPython minor version share its extension ABI.
    # Load before tkinter: replacing its backend after a Tk window exists is unsafe.
    if sys.platform.startswith("linux") and "_tkinter" not in sys.modules:
        version = f"python{sys.version_info.major}.{sys.version_info.minor}"
        suffix = sysconfig.get_config_var("EXT_SUFFIX")
        native = pathlib.Path("/usr/lib") / version / "lib-dynload" / f"_tkinter{suffix}"
        if native.is_file():
            spec = importlib.util.spec_from_file_location("_tkinter", native)
            try:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            except (ImportError, OSError):
                pass  # The interpreter's own Tk remains the fallback on other installations.
            else:
                sys.modules["_tkinter"] = module
    return importlib.import_module("tkinter")


tk = _desktop_tk()
# Import submodules only after selecting the native extension.
font = importlib.import_module("tkinter.font")
ttk = importlib.import_module("tkinter.ttk")
filedialog = importlib.import_module("tkinter.filedialog")
scrolledtext = importlib.import_module("tkinter.scrolledtext")


def configure_fonts(root):
    families = set(font.families(root))
    body = next(
        (
            name
            for name in (
                "Noto Sans CJK SC",
                "Microsoft YaHei UI",
                "PingFang SC",
                "WenQuanYi Micro Hei",
                "Noto Sans",
                "DejaVu Sans",
            )
            if name in families
        ),
        "TkDefaultFont",
    )
    code = next(
        (name for name in ("DejaVu Sans Mono", "Noto Sans Mono", "Consolas", "Menlo") if name in families), body
    )
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont", "TkTooltipFont"):
        font.nametofont(name, root=root).configure(family=body, size=10)
    font.nametofont("TkFixedFont", root=root).configure(family=code, size=9)
    root.option_add("*TCombobox*Listbox.font", "TkTextFont")
    return body
