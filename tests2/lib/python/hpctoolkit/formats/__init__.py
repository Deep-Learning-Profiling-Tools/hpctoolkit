## * BeginRiceCopyright *****************************************************
##
## $HeadURL$
## $Id$
##
## --------------------------------------------------------------------------
## Part of HPCToolkit (hpctoolkit.org)
##
## Information about sources of support for research and development of
## HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
## --------------------------------------------------------------------------
##
## Copyright ((c)) 2022-2022, Rice University
## All rights reserved.
##
## Redistribution and use in source and binary forms, with or without
## modification, are permitted provided that the following conditions are
## met:
##
## * Redistributions of source code must retain the above copyright
##   notice, this list of conditions and the following disclaimer.
##
## * Redistributions in binary form must reproduce the above copyright
##   notice, this list of conditions and the following disclaimer in the
##   documentation and/or other materials provided with the distribution.
##
## * Neither the name of Rice University (RICE) nor the names of its
##   contributors may be used to endorse or promote products derived from
##   this software without specific prior written permission.
##
## This software is provided by RICE and contributors "as is" and any
## express or implied warranties, including, but not limited to, the
## implied warranties of merchantability and fitness for a particular
## purpose are disclaimed. In no event shall RICE or contributors be
## liable for any direct, indirect, incidental, special, exemplary, or
## consequential damages (including, but not limited to, procurement of
## substitute goods or services; loss of use, data, or profits; or
## business interruption) however caused and on any theory of liability,
## whether in contract, strict liability, or tort (including negligence
## or otherwise) arising in any way out of the use of this software, even
## if advised of the possibility of such damage.
##
## ******************************************************* EndRiceCopyright *

import tarfile
import tempfile
from pathlib import Path, PurePath

import ruamel.yaml

from . import base, v4

__all__ = ["from_path", "from_path_extended"]

dir_classes: tuple[base.DatabaseBase, ...] = (v4.Database,)
file_classes: tuple[base.DatabaseFile, ...] = (
    v4.metadb.MetaDB,
    v4.profiledb.ProfileDB,
    v4.cctdb.ContextDB,
    v4.tracedb.TraceDB,
)


def from_path(src: Path) -> base.DatabaseBase | base.DatabaseFile | None:
    """Open a file/directory of any of the supported formats. Returns the object-form of the input,
    or None if it does not appear to be a supported format."""

    if src.is_dir():
        # Presume it's a database directory
        for cls in dir_classes:
            try:
                return cls.from_dir(src)
            except (base.InvalidFormatError, base.IncompatibleFormatError):
                pass
    elif src.is_file():
        # Presume it's a loose data file
        with open(src, "rb") as srcf:
            for cls in file_classes:
                try:
                    return cls.from_file(srcf)
                except (base.InvalidFormatError, base.IncompatibleFormatError):
                    pass
    elif not src.exists():
        raise FileNotFoundError(src)
    return None


def _iter_deep_dir(path: Path):
    yield path
    while path.is_dir():
        children = [path.iterdir()]
        if len(children) == 1:
            path = children[0]
            yield path


def from_path_extended(
    src: Path, *, subdir: PurePath = None
) -> base.DatabaseBase | base.DatabaseFile | None:
    """Extension of from_path that also supports some more obscure but convenient formats.

    Supports:
     - Compressed (xz, gzip, bzip2) and uncompressed tarballs containing a single database dir/file
     - Tarballs containing multiple database directories (when subdir is provided)
     - YAML databases as generated by hpctoolkit.formats.print
    """

    # Attempt 1: It's a normal database/file
    result = from_path(src)
    if result is not None:
        return result

    if src.is_file():
        # Attempt 2: It's a tarball containing a database directory or file
        try:
            with tempfile.TemporaryDirectory() as d:
                pd = Path(d)
                with tarfile.open(src, mode="r:*") as tf:
                    tf.extractall(pd)

                if subdir is None:
                    for attempt in _iter_deep_dir(pd):
                        result = from_path(attempt)
                        if result is not None:
                            return result
                else:
                    result = from_path(pd / subdir)
                    if result is not None:
                        return result
        except tarfile.TarError:
            pass

        # Attempt 3: It's a YAML file containing a serialized database or otherwise
        try:
            with open(src, encoding="utf-8") as f:
                result = ruamel.yaml.YAML(typ="safe").load(f)
                if isinstance(result, base.DatabaseBase | base.DatabaseFile):
                    return result
        except ruamel.yaml.YAMLError:
            pass

    # All attempts failed, give up
    return None
