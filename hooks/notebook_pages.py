# Copyright (C) 2026, Tactics2D Authors. Released under the GNU GPLv3.
# SPDX-License-Identifier: GPL-3.0-or-later

"""Keep notebook files working as pages next to the i18n plugin."""

import os
from pathlib import Path

from mkdocs.config.defaults import MkDocsConfig
from mkdocs.plugins import event_priority
from mkdocs.structure.files import Files


def _is_a_page() -> bool:
    """Report every restored notebook as a documentation page."""
    return True


@event_priority(-200)
def on_files(files: Files, config: MkDocsConfig) -> Files:
    """Restore the notebook pages that the i18n plugin rebuilds as plain files.

    mkdocs-static-i18n replaces every file with a fresh File instance, which drops the
    wrapper mkdocs-jupyter relies on to make .ipynb a documentation page. Without it
    MkDocs copies the raw notebook instead of rendering it, and the i18n plugin stops
    giving the notebook the language directory it gives to pages. Both are restored
    here, after the i18n plugin (priority -100) has built the file collection.
    """
    jupyter = config.plugins.get("mkdocs-jupyter")
    i18n = config.plugins.get("i18n")
    if jupyter is None or i18n is None:
        return files

    if i18n.current_language == i18n.default_language:
        locale_dir = ""
    else:
        locale_dir = i18n.current_language

    for file in files:
        if file.is_documentation_page() or not jupyter.should_include(file):
            continue
        if locale_dir:
            # Same relocation the i18n plugin applies to the pages of a language.
            file.dest_path = Path(locale_dir) / file.dest_path
            file.abs_dest_path = os.path.normpath(os.path.join(config.site_dir, file.dest_path))
            file.url = file._get_url(config.use_directory_urls)
        file.is_documentation_page = _is_a_page
    return files
