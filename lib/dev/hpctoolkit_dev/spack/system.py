import collections.abc
import enum
import re
import shutil
import typing
from pathlib import Path

import click
from spiqa.syntax import Version

from .abc import CompilerBase


class OSClass(enum.Enum):
    DebianLike = "Debian/Ubuntu"
    RedHatLike = "Fedora/RHEL"
    SUSELeap = "OpenSUSE Leap"


_TRANSLATIONS_RH: collections.abc.Mapping[str, str | None] = {
    "bzip2": None,
    "ccache": None,
    "diffutils": None,
    "eatmydata": None,
    "file": None,
    "g++": None,
    "gcc": None,
    "git-lfs": None,
    "git": None,
    "gnupg2": None,
    "gzip": None,
    "make": None,
    "patch": None,
    "tar": None,
    "unzip": None,
    "xz-utils": "xz",
    "zstd": None,
}
_TRANSLATIONS_SUSE: collections.abc.Mapping[str, str | None] = _TRANSLATIONS_RH


def translate(pkg: str, target: OSClass) -> str:
    """Translate the Debian/Ubuntu name for an OS package into the name for a target OS."""
    match target:
        case OSClass.DebianLike:
            return pkg
        case OSClass.RedHatLike:
            if pkg in _TRANSLATIONS_RH:
                return _TRANSLATIONS_RH[pkg] or pkg
        case OSClass.SUSELeap:
            if pkg in _TRANSLATIONS_SUSE:
                return _TRANSLATIONS_SUSE[pkg] or pkg
    click.echo(f"WARNING: Unrecognized package {pkg} for target OS {target}, not translating")
    return pkg


@typing.final
class SystemCompiler(CompilerBase):
    """Compiler that represents a compiler installed on the system, referred to via a simple syntax."""

    CC_TO_CPP: typing.ClassVar[dict] = {
        "gcc": "g++",
        "clang": "clang++",
    }

    def __init__(self, basename: str) -> None:
        """Create a new SystemCompiler from the basename of the compiler."""
        match = re.fullmatch(r"([^=]+?)(?:-(\d+))?(?:=.*)?", basename)
        if not match:
            raise ValueError(f"Invalid syntax for compiler basename: {basename}")

        self._name, suffix = match[1], ("-" + match[2] if match[2] else "")
        if self._name not in self.CC_TO_CPP:
            raise ValueError(f"Unrecognized compiler name: {self._name!r}")

        self._cc = shutil.which(self._name + suffix)
        self._cpp = shutil.which(self.CC_TO_CPP[self._name] + suffix)

        self._basename = basename
        self._raw_version = int(match[2]) if match[2] else None
        self._version = Version(match[2]) if match[2] else Version("99")

    def os_packages(self, os: OSClass) -> set[str]:
        rv = self._raw_version
        match os:
            case OSClass.DebianLike:
                match self.name:
                    case "gcc":
                        return {f"gcc-{rv:d}", f"g++-{rv:d}"} if rv is not None else {"gcc", "g++"}
                    case "clang":
                        return (
                            {f"clang-{rv:d}", f"clang++-{rv:d}"}
                            if rv is not None
                            else {"clang", "clang++"}
                        )

            case OSClass.SUSELeap:
                match self.name:
                    case "gcc":
                        return (
                            {f"gcc{rv:d}", f"gcc{rv:d}-c++"}
                            if rv is not None
                            else {"gcc", "gcc-c++"}
                        )
                    case "clang":
                        return {f"clang{rv:d}"} if rv is not None else {"clang"}

            case OSClass.RedHatLike:
                if rv is not None:
                    raise click.ClickException(f"{os} does not support versioned compilers: {self}")

                match self.name:
                    case "gcc":
                        return {"gcc", "gcc-c++"}
                    case "clang":
                        return {"clang"}

        raise click.ClickException(f"Unsupported compiler for {os}: {self.name}")

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> Version:
        return self._version

    def __str__(self) -> str:
        return self._basename

    def __bool__(self) -> bool:
        return bool(self._cc and self._cpp)

    @property
    def cc(self) -> Path:
        if not self._cc:
            raise AttributeError("cc")
        return Path(self._cc)

    @property
    def cpp(self) -> Path:
        if not self._cpp:
            raise AttributeError("cpp")
        return Path(self._cpp)
