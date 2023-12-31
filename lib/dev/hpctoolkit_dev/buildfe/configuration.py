import atexit
import collections
import collections.abc
import contextlib
import dataclasses
import functools
import itertools
import os
import re
import shutil
import stat
import tempfile
import textwrap
import typing
from pathlib import Path

from elftools.elf.elffile import ELFFile  # type: ignore[import]

from ..meson import MesonMachineFile
from .logs import FgColor, colorize


class UnsatisfiableSpecError(Exception):
    """Exception raised when the given variant is unsatisfiable."""

    def __init__(self, missing):
        super().__init__(f"missing definition for argument {missing}")
        self.missing = missing


class ImpossibleSpecError(Exception):
    """Exception raised when the given variant is impossible."""

    def __init__(self, a, b):
        super().__init__(f'conflict between "{a}" and "{b}"')
        self.a, self.b = a, b


class Compilers:
    """State needed to derive configure-time arguments, based on simple configuration files."""

    def __init__(self, cc: str | None = None):
        self.configs: list[tuple[Path | None, str]] = []
        self._compiler: tuple[str, str] | None = None
        self._compiler_wrappers: tuple[Path, Path] | None = None
        if cc is not None:
            if not cc or cc[0] == "-":
                raise ValueError(cc)

            cc = cc.split("=")[0]  # Anything after an = is considered a comment
            mat = re.fullmatch("([^-]+)(-?.*)", cc)
            assert mat
            match mat.group(1):
                case "gcc":
                    self._compiler = "gcc" + mat.group(2), "g++" + mat.group(2)
                case "clang":
                    self._compiler = "clang" + mat.group(2), "clang++" + mat.group(2)
                case _:
                    raise ValueError(cc)

    @functools.cached_property
    def raw_compilers(self) -> tuple[str, str]:
        if self._compiler is not None:
            return self._compiler
        cc, cxx = os.environ.get("CC", "cc"), os.environ.get("CXX", "c++")
        if not shutil.which(cc) or not shutil.which(cxx):
            raise RuntimeError("Unable to guess system compilers, cc/c++ not present!")
        return cc, cxx

    @functools.cached_property
    def custom_compilers(self) -> tuple[str, str] | None:
        ccache = shutil.which("ccache")
        if self._compiler is not None or ccache:
            raw_cc, raw_cxx = self.raw_compilers
            if ccache:
                ccdir = Path(tempfile.mkdtemp(".ccwrap"))
                atexit.register(shutil.rmtree, ccdir, ignore_errors=True)
                cc, cxx = ccdir / "cc", ccdir / "cxx"
                with open(cc, "w", encoding="utf-8") as f:
                    f.write(f'#!/bin/sh\nexec {ccache} {raw_cc} "$@"')
                with open(cxx, "w", encoding="utf-8") as f:
                    f.write(f'#!/bin/sh\nexec {ccache} {raw_cxx} "$@"')
                cc.chmod(cc.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                cxx.chmod(cxx.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
                return str(cc), str(cxx)
            return raw_cc, raw_cxx
        return None

    @functools.cached_property
    def compilers(self) -> tuple[str, str]:
        if self.custom_compilers is not None:
            return self.custom_compilers
        return self.raw_compilers


class _SymbolManifest:
    def __init__(self, symbols: frozenset[str], *, allow_extra: bool | re.Pattern = False):
        self.symbols = symbols
        self.allow_extra: re.Pattern | None = None
        if allow_extra:
            self.allow_extra = re.compile(r".*") if allow_extra is True else allow_extra

    def check(self, path: Path) -> str | None:
        with open(path, "rb") as f, ELFFile(f) as file:
            got = {
                s.name
                for seg in file.iter_segments("PT_DYNAMIC")
                for s in seg.iter_symbols()
                if s["st_info"].type == "STT_FUNC"
                and s["st_info"].bind in ("STB_GLOBAL", "STB_WEAK")
                and s["st_value"] != 0
            }
            missing, unexpected = self.symbols - got, got - self.symbols
            if self.allow_extra:
                unexpected = {s for s in unexpected if not self.allow_extra.fullmatch(s)}

            lines: list[str] = []
            if missing:
                lines.append("The following symbols were expected but not found:")
                for sym in missing:
                    lines.append(f"  {sym!r},")
            if unexpected:
                lines.append("The following symbols were unexpected:")
                for sym in unexpected:
                    lines.append(f"  {sym!r},")
            return "\n".join(lines) if lines else None


class _ManifestFile:
    def __init__(self, path, *, symbols: _SymbolManifest | None = None):
        self.path = Path(path)
        self.symbols = symbols

    def check(self, installdir) -> tuple[set[Path], set[Path | tuple[Path, str]]]:
        if not (Path(installdir) / self.path).is_file():
            return set(), {self.path}
        if (
            self.symbols is not None
            and (symerr := self.symbols.check(Path(installdir) / self.path)) is not None
        ):
            return set(), {
                (self.path, f"Symbol manifest failed to match:\n{textwrap.indent(symerr, '  ')}")
            }
        return {self.path}, set()


class _ManifestLib(_ManifestFile):
    def __init__(self, path, target, *aliases, symbols: _SymbolManifest | None = None):
        super().__init__(path)
        self.target = str(target)
        self.aliases = [str(a) for a in aliases]
        self.symbols = symbols

    def check(self, installdir) -> tuple[set[Path], set[Path | tuple[Path, str]]]:
        installdir = Path(installdir)
        found, missing = set(), set()

        target = installdir / self.path
        target = target.with_name(target.name + self.target)
        if not target.is_file():
            missing.add(target.relative_to(installdir))
        if target.is_symlink():
            missing.add(
                (target.relative_to(installdir), f"Unexpected symlink to {os.readlink(target)}")
            )
        if self.symbols is not None and (symerr := self.symbols.check(target)) is not None:
            missing.add(
                (
                    target.relative_to(installdir),
                    f"Symbol manifest failed to match:\n{textwrap.indent(symerr, '  ')}",
                )
            )

        for a in self.aliases:
            alias = installdir / self.path
            alias = alias.with_name(alias.name + a)
            if not alias.is_file():
                missing.add(alias.relative_to(installdir))
                continue
            if not alias.is_absolute():
                missing.add((alias.relative_to(installdir), "Not a symlink"))
                continue

            targ = Path(os.readlink(alias))
            if len(targ.parts) > 1:
                missing.add(
                    (alias.relative_to(installdir), "Invalid symlink, must point to sibling file")
                )
                continue
            if targ.name != target.name:
                missing.add(
                    (alias.relative_to(installdir), f"Invalid symlink, must point to {target.name}")
                )
                continue

            found.add(alias.relative_to(installdir))

        return found, missing


class _ManifestExtLib(_ManifestFile):
    def __init__(self, path, main_suffix, *suffixes, symbols: _SymbolManifest | None = None):
        super().__init__(path)
        self.main_suffix = str(main_suffix)
        self.suffixes = [str(s) for s in suffixes]
        self.symbols = symbols

    def check(self, installdir) -> tuple[set[Path], set[Path | tuple[Path, str]]]:
        installdir = Path(installdir)
        common = installdir / self.path
        found = set()

        main_path = common.with_name(common.name + self.main_suffix)
        if not main_path.is_file():
            return set(), {main_path.relative_to(installdir)}
        if self.symbols is not None and (symerr := self.symbols.check(main_path)) is not None:
            return set(), {
                (
                    main_path.relative_to(installdir),
                    f"Symbol manifest failed to match:\n{textwrap.indent(symerr, '  ')}",
                )
            }
        found.add(main_path.relative_to(installdir))

        missing: set[Path | tuple[Path, str]] = set()
        for path in common.parent.iterdir():
            if path.name.startswith(common.name) and path != main_path:
                name = path.name[len(common.name) :]
                if any(re.match(s, name) for s in self.suffixes):
                    if (
                        self.symbols is not None
                        and (symerr := self.symbols.check(path)) is not None
                    ):
                        missing.add(
                            (
                                path.relative_to(installdir),
                                f"Symbol manifest failed to match:\n{textwrap.indent(symerr, '  ')}",
                            )
                        )
                        continue
                    found.add(path.relative_to(installdir))

        return found, missing


class Manifest:
    """Representation of an install manifest."""

    SYMBOLS_LIBHPCRUN = frozenset(
        {
            "__sysv_signal",
            "debug_flag_dump",
            "debug_flag_get",
            "debug_flag_init",
            "debug_flag_set",
            "hpctoolkit_sampling_is_active",
            "hpctoolkit_sampling_start",
            "hpctoolkit_sampling_start_",
            "hpctoolkit_sampling_start__",
            "hpctoolkit_sampling_stop",
            "hpctoolkit_sampling_stop_",
            "hpctoolkit_sampling_stop__",
            "messages_donothing",
            "messages_fini",
            "messages_init",
            "messages_logfile_create",
            "messages_logfile_fd",
            "monitor_at_main",
            "monitor_begin_process_exit",
            "monitor_fini_process",
            "monitor_fini_thread",
            "monitor_init_mpi",
            "monitor_init_process",
            "monitor_init_thread",
            "monitor_mpi_pre_init",
            "monitor_post_fork",
            "monitor_pre_fork",
            "monitor_reset_stacksize",
            "monitor_start_main_init",
            "monitor_thread_post_create",
            "monitor_thread_pre_create",
            "ompt_start_tool",
            "poll",
            "ppoll",
            "pselect",
            "select",
        }
    )
    SYMBOLS_LIBHPCRUN_ROCM = frozenset(
        {
            "OnUnloadTool",
            "OnLoadToolProp",
        }
    )
    SYMBOLS_LIBHPCRUN_DLMOPEN = frozenset(
        {
            "dlmopen",
        }
    )
    SYMBOLS_LIBHPCRUN_FAKE_AUDIT = frozenset(
        {
            "dlclose",
            "dlmopen",
            "dlopen",
            "hpcrun_init_fake_auditor",
        }
    )
    SYMBOLS_LIBHPCRUN_GA = frozenset(
        {
            "pnga_acc",
            "pnga_brdcst",
            "pnga_create",
            "pnga_create_handle",
            "pnga_get",
            "pnga_gop",
            "pnga_nbacc",
            "pnga_nbget",
            "pnga_nbput",
            "pnga_nbwait",
            "pnga_put",
            "pnga_sync",
        }
    )
    SYMBOLS_LIBHPCRUN_GPROF = frozenset(
        {
            "__monstartup",
            "_mcleanup",
            "_mcount",
            "mcount",
        }
    )
    SYMBOLS_LIBHPCRUN_IO = frozenset(
        {
            "fread",
            "fwrite",
            "read",
            "write",
        }
    )
    SYMBOLS_LIBHPCRUN_MEMLEAK = frozenset(
        {
            "calloc",
            "free",
            "malloc",
            "memalign",
            "posix_memalign",
            "realloc",
            "valloc",
        }
    )
    SYMBOLS_LIBHPCRUN_PTHREAD = frozenset(
        {
            "override_lookup",
            "override_lookupv",
            "pthread_cond_broadcast",
            "pthread_cond_signal",
            "pthread_cond_timedwait",
            "pthread_cond_wait",
            "pthread_mutex_lock",
            "pthread_mutex_timedlock",
            "pthread_mutex_unlock",
            "pthread_spin_lock",
            "pthread_spin_unlock",
            "sched_yield",
            "sem_post",
            "sem_timedwait",
            "sem_wait",
            "tbb_stats",
        }
    )
    SYMBOLS_LIBMONITOR = frozenset(
        {
            "MPI_Comm_rank",
            "MPI_Finalize",
            "MPI_Init",
            "MPI_Init_thread",
            "PMPI_Comm_rank",
            "PMPI_Finalize",
            "PMPI_Init",
            "PMPI_Init_thread",
            "_Exit",
            "__libc_start_main",
            "_exit",
            "execl",
            "execle",
            "execlp",
            "execv",
            "execve",
            "execvp",
            "exit",
            "fork",
            "monitor_adjust_stack_size",
            "monitor_at_main",
            "monitor_begin_process_exit",
            "monitor_begin_process_fcn",
            "monitor_block_shootdown",
            "monitor_broadcast_signal",
            "monitor_disable_new_threads",
            "monitor_dlclose",
            "monitor_dlopen",
            "monitor_dlsym",
            "monitor_early_init",
            "monitor_enable_new_threads",
            "monitor_end_library_fcn",
            "monitor_end_process_fcn",
            "monitor_fini_library",
            "monitor_fini_mpi",
            "monitor_fini_process",
            "monitor_fini_thread",
            "monitor_fork_init",
            "monitor_get_addr_main",
            "monitor_get_addr_thread_start",
            "monitor_get_main_args",
            "monitor_get_main_tn",
            "monitor_get_new_thread_info",
            "monitor_get_thread_num",
            "monitor_get_tn",
            "monitor_get_user_data",
            "monitor_in_main_start_func_narrow",
            "monitor_in_main_start_func_wide",
            "monitor_in_start_func_narrow",
            "monitor_in_start_func_wide",
            "monitor_init_library",
            "monitor_init_mpi",
            "monitor_init_process",
            "monitor_init_thread",
            "monitor_init_thread_support",
            "monitor_initialize",
            "monitor_is_threaded",
            "monitor_library_fini_destructor",
            "monitor_library_init_constructor",
            "monitor_main",
            "monitor_mpi_comm_rank",
            "monitor_mpi_comm_size",
            "monitor_mpi_fini_count",
            "monitor_mpi_init_count",
            "monitor_mpi_post_fini",
            "monitor_mpi_pre_init",
            "monitor_post_dlclose",
            "monitor_post_fork",
            "monitor_pre_dlopen",
            "monitor_pre_fork",
            "monitor_real_abort",
            "monitor_real_dlclose",
            "monitor_real_dlopen",
            "monitor_real_execve",
            "monitor_real_exit",
            "monitor_real_fork",
            "monitor_real_pthread_sigmask",
            "monitor_real_sigprocmask",
            "monitor_real_system",
            "monitor_remove_client_signals",
            "monitor_reset_stacksize",
            "monitor_reset_thread_list",
            "monitor_set_mpi_size_rank",
            "monitor_set_size_rank",
            "monitor_shootdown_signal",
            "monitor_sigaction",
            "monitor_signal_init",
            "monitor_signal_list_string",
            "monitor_sigset_string",
            "monitor_sigwait_handler",
            "monitor_stack_bottom",
            "monitor_start_main_init",
            "monitor_thread_post_create",
            "monitor_thread_pre_create",
            "monitor_thread_shootdown",
            "monitor_unblock_shootdown",
            "monitor_unwind_process_bottom_frame",
            "monitor_unwind_thread_bottom_frame",
            "monitor_wrap_main",
            "mpi_comm_rank",
            "mpi_comm_rank_",
            "mpi_comm_rank__",
            "mpi_finalize",
            "mpi_finalize_",
            "mpi_finalize__",
            "mpi_init",
            "mpi_init_",
            "mpi_init__",
            "mpi_init_thread",
            "mpi_init_thread_",
            "mpi_init_thread__",
            "pmpi_comm_rank",
            "pmpi_comm_rank_",
            "pmpi_comm_rank__",
            "pmpi_finalize",
            "pmpi_finalize_",
            "pmpi_finalize__",
            "pmpi_init",
            "pmpi_init_",
            "pmpi_init__",
            "pmpi_init_thread",
            "pmpi_init_thread_",
            "pmpi_init_thread__",
            "pthread_create",
            "pthread_exit",
            "pthread_sigmask",
            "sigaction",
            "signal",
            "sigprocmask",
            "sigtimedwait",
            "sigwait",
            "sigwaitinfo",
            "system",
            "vfork",
        }
    )

    def __init__(self, *, mpi: bool, rocm: bool):
        """Given a set of variant-keywords, determine the install manifest as a list of ManifestFiles."""

        def so(n):
            return r"\.so" + r"\.\d+" * n.__index__()

        hpcrun_symbols = self.SYMBOLS_LIBHPCRUN
        if rocm:
            hpcrun_symbols |= self.SYMBOLS_LIBHPCRUN_ROCM
        hpcrun_extra = re.compile(r"^hpcrun_.+$")

        self.files = [
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_atomic-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_atomic", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_chrono", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_date_time-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_date_time", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_filesystem-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_filesystem", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_graph-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_graph", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_regex-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_regex", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_system-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_system", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_thread-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_thread", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_timer-mt", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libboost_timer", ".so", so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libbz2", ".so", so(1), so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libcommon", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libdw", ".so", so(1), r"-\d+\.\d+\.so"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libdynDwarf", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libdynElf", ".so", so(1), r"-\d+\.\d+\.so"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libelf", ".so", so(1), r"-\d+\.\d+\.so"),
            _ManifestExtLib(
                "lib/hpctoolkit/ext-libs/libinstructionAPI", ".so", so(1), r"-\d+\.\d+\.so"
            ),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libmonitor_wrap", ".a"),
            _ManifestExtLib(
                "lib/hpctoolkit/ext-libs/libmonitor",
                ".so",
                ".so.0",
                ".so.0.0.0",
                symbols=_SymbolManifest(self.SYMBOLS_LIBMONITOR, allow_extra=True),
            ),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libparseAPI", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libpfm", ".so", so(1), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libsymtabAPI", ".so", so(2), so(3)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libtbb", ".so", so(1)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libtbbmalloc_proxy", ".so", so(1)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libtbbmalloc", ".so", so(1)),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libxerces-c", ".a"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libxerces-c", ".so", r"-\d+.\d+\.so"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libz", ".a"),
            _ManifestExtLib("lib/hpctoolkit/ext-libs/libz", ".so", so(1), so(3)),
            _ManifestFile("bin/hpclink"),
            _ManifestFile("bin/hpcprof"),
            _ManifestFile("bin/hpcrun"),
            _ManifestFile("bin/hpcstruct"),
            _ManifestFile("include/hpctoolkit.h"),
            _ManifestFile("lib/hpctoolkit/hash-file"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_audit.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_audit.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_dlmopen.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_dlmopen.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_fake_audit.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_fake_audit.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_ga.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_ga.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_gprof.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_gprof.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_io.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_io.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_memleak.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_memleak.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_pthread.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_pthread.la"),
            _ManifestFile("lib/hpctoolkit/libhpcrun_wrap.a"),
            _ManifestFile("lib/hpctoolkit/libhpcrun.o"),
            _ManifestFile(
                "lib/hpctoolkit/libhpcrun.so",
                symbols=_SymbolManifest(hpcrun_symbols, allow_extra=hpcrun_extra),
            ),
            # XXX: Why is there no libhpcrun.so.0.0.0 and libhpcrun.la?
            _ManifestFile("lib/hpctoolkit/libhpctoolkit.a"),
            _ManifestFile("lib/hpctoolkit/libhpctoolkit.la"),
            _ManifestFile("lib/hpctoolkit/plugins/ga"),
            _ManifestFile("lib/hpctoolkit/plugins/io"),
            _ManifestFile("lib/hpctoolkit/plugins/memleak"),
            _ManifestFile("lib/hpctoolkit/plugins/pthread"),
            # XXX: Why is there no gprof?
            _ManifestFile("libexec/hpctoolkit/config.guess"),
            _ManifestFile("libexec/hpctoolkit/dotgraph-bin"),
            _ManifestFile("libexec/hpctoolkit/dotgraph"),
            _ManifestFile("libexec/hpctoolkit/hpcfnbounds"),
            _ManifestFile("libexec/hpctoolkit/hpcguess"),
            _ManifestFile("libexec/hpctoolkit/hpclog"),
            _ManifestFile("libexec/hpctoolkit/hpcplatform"),
            _ManifestFile("libexec/hpctoolkit/hpcproftt-bin"),
            _ManifestFile("libexec/hpctoolkit/hpcproftt"),
            _ManifestFile("libexec/hpctoolkit/hpcsummary"),
            _ManifestFile("libexec/hpctoolkit/hpctracedump"),
            _ManifestFile("share/doc/hpctoolkit/FORMATS.md"),
            _ManifestFile("share/doc/hpctoolkit/METRICS.yaml"),
            _ManifestFile("share/doc/hpctoolkit/LICENSE"),
            _ManifestFile("share/doc/hpctoolkit/man/hpclink.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcprof.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcproftt.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcrun.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcstruct.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpctoolkit.html"),
            _ManifestFile("share/doc/hpctoolkit/man/hpcviewer.html"),
            _ManifestFile("share/doc/hpctoolkit/manual/HPCToolkit-users-manual.pdf"),
            _ManifestFile("share/doc/hpctoolkit/README.Acknowledgments"),
            _ManifestFile("share/doc/hpctoolkit/README.Install"),
            _ManifestFile("share/doc/hpctoolkit/README.md"),
            _ManifestFile("share/doc/hpctoolkit/README.ReleaseNotes"),
            _ManifestFile("share/hpctoolkit/dtd/hpc-experiment.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/hpc-structure.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/hpcprof-config.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsa.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsb.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsc.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsn.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamso.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isoamsr.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isobox.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isocyr1.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isocyr2.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isodia.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isogrk3.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isolat1.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isolat2.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isomfrk.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isomopf.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isomscr.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isonum.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isopub.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/isotech.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/mathml.dtd"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/mmlalias.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/mmlextra.ent"),
            _ManifestFile("share/hpctoolkit/dtd/mathml/xhtml1-transitional-mathml.dtd"),
            _ManifestFile("share/man/man1/hpclink.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcprof.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcproftt.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcrun.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcstruct.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpctoolkit.1hpctoolkit"),
            _ManifestFile("share/man/man1/hpcviewer.1hpctoolkit"),
            _ManifestLib("lib/hpctoolkit/libhpcrun_audit.so", ".0.0.0", ".0", ""),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_dlmopen.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_DLMOPEN),
            ),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_fake_audit.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_FAKE_AUDIT),
            ),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_ga.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_GA),
            ),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_gprof.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_GPROF),
            ),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_io.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_IO),
            ),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_memleak.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_MEMLEAK),
            ),
            _ManifestLib(
                "lib/hpctoolkit/libhpcrun_pthread.so",
                ".0.0.0",
                ".0",
                "",
                symbols=_SymbolManifest(self.SYMBOLS_LIBHPCRUN_PTHREAD),
            ),
            _ManifestLib("lib/hpctoolkit/libhpctoolkit.so", ".0.0.0", ".0", ""),
        ]

        if mpi:
            self.files += [
                _ManifestFile("bin/hpcprof-mpi"),
            ]

    def check(self, installdir: Path) -> tuple[int, int]:
        """Scan an install directory and compare against the expected manifest. Prints the results
        of the checks to the log. Return the counts of missing and unexpected files.
        """
        # First derive the full listing of actually installed files
        listing = set()
        for root, _, files in os.walk(installdir):
            for filename in files:
                listing.add((Path(root) / filename).relative_to(installdir))

        # Then match these files up with the results we found
        n_unexpected = 0
        n_uninstalled = 0
        warnings: list[str] = []
        errors: list[str] = []
        for f in self.files:
            found, not_found = f.check(installdir)
            warnings.extend(f"+ {fn.as_posix()}" for fn in found - listing)
            n_unexpected += len(found - listing)
            listing -= found
            for fn in not_found:
                if isinstance(fn, tuple):
                    fn2, msg = fn
                    errors.append(f"! {fn2.as_posix()}\n  ^ {textwrap.indent(msg, '    ')}")
                else:
                    errors.append(f"- {fn.as_posix()}")
            n_uninstalled += len(not_found)

        # Print out the warnings and then the errors, with colors
        with colorize(FgColor.warning):
            for hunk in warnings:
                print(hunk)
        with colorize(FgColor.error):
            for hunk in errors:
                print(hunk)

        return n_uninstalled, n_unexpected


@dataclasses.dataclass(kw_only=True, frozen=True)
class ConcreteSpecification:
    """Point in the build configuration space, represented as a series of boolean-valued variants."""

    mpi: bool
    debug: bool
    valgrind_debug: bool
    papi: bool
    python: bool
    opencl: bool
    cuda: bool
    rocm: bool
    level0: bool
    gtpin: bool

    def __post_init__(self, **_kwargs):
        if self.gtpin and not self.level0:
            raise ImpossibleSpecError("+gtpin", "~level0")
        if self.valgrind_debug and not self.debug:
            raise ImpossibleSpecError("+valgrind_debug", "~debug")

    @classmethod
    def all_possible(cls) -> collections.abc.Iterable["ConcreteSpecification"]:
        """List all possible ConcreteSpecifications in the configuration space."""
        dims = [
            # (variant, values...), in order from slowest- to fastest-changing
            ("gtpin", False, True),
            ("level0", False, True),
            ("rocm", False, True),
            ("cuda", False, True),
            ("opencl", False, True),
            ("python", False, True),
            ("papi", True, False),
            ("valgrind_debug", False, True),
            ("debug", True, False),
            ("mpi", False, True),
        ]
        for point in itertools.product(*[[(x[0], val) for val in x[1:]] for x in dims]):
            with contextlib.suppress(ImpossibleSpecError):
                yield cls(**dict(point))

    @classmethod
    def variants(cls) -> tuple[str, ...]:
        """List the names of all the available variants."""
        return tuple(f.name for f in dataclasses.fields(cls))

    def __str__(self) -> str:
        return " ".join(f"+{n}" if getattr(self, n) else f"~{n}" for n in self.variants())

    @property
    def without_spaces(self) -> str:
        return "".join(f"+{n}" if getattr(self, n) else f"~{n}" for n in self.variants())


class Specification:
    """Subset of the build configuration space."""

    @dataclasses.dataclass
    class _Condition:
        min_enabled: int
        max_enabled: int

    _concrete_cls: typing.ClassVar[type[ConcreteSpecification]] = ConcreteSpecification
    _clauses: dict[tuple[str, ...], _Condition] | bool

    def _parse_long(self, valid_variants: frozenset[str], clause: str) -> set[str]:
        mat = re.fullmatch(r"\(([\w\s]+)\)\[([+~><=\d\s]+)\]", clause)
        if not mat:
            raise ValueError(f"Invalid clause: {clause!r}")

        variants = mat.group(1).split()
        if not variants:
            raise ValueError(f"Missing variants listing: {clause!r}")
        for v in variants:
            if v not in valid_variants:
                raise ValueError("Invalid variant: {v}")
        variants = tuple(sorted(variants))

        assert isinstance(self._clauses, dict)
        cond = self._clauses.setdefault(variants, self._Condition(0, len(variants)))
        for c in mat.group(2).split():
            cmat = re.fullmatch(r"([+~])([><=])(\d+)", c)
            if not cmat:
                raise ValueError(f"Invalid conditional expression: {c!r}")
            match cmat.group(1):
                case "+":
                    n, op = int(cmat.group(3)), cmat.group(2)
                case "~":
                    # ~>N is +<(V-N), etc.
                    n = len(variants) - int(cmat.group(3))
                    op = {">": "<", "<": ">", "=": "="}[cmat.group(2)]
                case _:
                    raise AssertionError()
            for base_op in {">": ">", "<": "<", "=": "<>"}[op]:
                match base_op:
                    case ">":
                        cond.min_enabled = max(cond.min_enabled, n)
                    case "<":
                        cond.max_enabled = min(cond.max_enabled, n)
                    case _:
                        raise AssertionError()
        return set(variants)

    def _parse_short(self, valid_variants: frozenset[str], clause: str) -> set[str]:
        mat = re.fullmatch(r"([+~])(\w+)", clause)
        if not mat:
            raise ValueError(f"Invalid clause: {clause!r}")

        match mat.group(1):
            case "+":
                cnt = 1
            case "~":
                cnt = 0
            case _:
                raise AssertionError()

        variant = mat.group(2)
        if variant not in valid_variants:
            raise ValueError(f"Invalid variant: {variant}")

        assert isinstance(self._clauses, dict)
        cond = self._clauses.setdefault((variant,), self._Condition(0, 1))
        cond.min_enabled = max(cond.min_enabled, cnt)
        cond.max_enabled = min(cond.max_enabled, cnt)
        return {variant}

    def _optimize(self):
        assert isinstance(self._clauses, dict)

        # Fold multi-variant clauses that have a single solution into single-variant clauses
        new_clauses: dict[tuple[str, ...], Specification._Condition] = {}
        for vrs, cond in self._clauses.items():
            assert len(vrs) > 0
            assert cond.min_enabled <= len(vrs)
            assert cond.max_enabled <= len(vrs)
            if (
                len(vrs) > 1
                and cond.min_enabled == cond.max_enabled
                and cond.min_enabled in (0, len(vrs))
            ):
                for v in vrs:
                    newcond = new_clauses.setdefault((v,), self._Condition(0, 1))
                    newcond.min_enabled = max(newcond.min_enabled, cond.min_enabled)
                    newcond.max_enabled = min(newcond.max_enabled, cond.max_enabled)
            else:
                new_clauses[vrs] = cond
        self._clauses = new_clauses

        # Prune clauses that trivially can't be satisfied or are always satisfied. For cleanliness.
        good_clauses = {}
        for vrs, cond in self._clauses.items():
            assert len(vrs) > 0
            assert cond.min_enabled <= len(vrs)
            assert cond.max_enabled <= len(vrs)
            if cond.min_enabled <= cond.max_enabled and (
                cond.min_enabled > 0 or cond.max_enabled < len(vrs)
            ):
                good_clauses[vrs] = cond
        self._clauses = good_clauses if good_clauses else False

    def __init__(  # noqa: C901
        self,
        spec: str,
        /,
        *,
        allow_blank: bool = False,
        allow_empty: bool = False,
        strict: bool = True,
    ):
        """Create a new Specification from the given specification string.

        The syntax follows this rough BNF grammar:
            spec := all | none | W { clause W }*
            clause := '!' variant
                      | flag variant
                      | '(' W variant { W variant }* W ')[' W condition { W condition }* W ']'
            flag := '+' | '~'
            condition := flag comparison N
            comparison := '>' | '<' | '='
            variant := # any variant listed in ConcreteSpecification.variants()
            N := # any base 10 number
            W := # any sequence of whitespace
        """
        valid_variants = frozenset(self._concrete_cls.variants())
        constrained_variants = set()

        # Parse the spec itself
        match spec.strip():
            case "all":
                self._clauses = True
            case "none":
                self._clauses = False
            case _:
                self._clauses = {}
                for token in re.split(r"(\([^)]+\)\[[^]]+\])|\s", spec):
                    if token is None or not token:
                        continue
                    if token[0] == "!":
                        if token[1:] not in valid_variants:
                            raise ValueError(f"Invalid variant: {token[1:]}")
                        constrained_variants.add(token[1:])
                    elif token[0] == "(":
                        constrained_variants |= self._parse_long(valid_variants, token)
                    else:
                        constrained_variants |= self._parse_short(valid_variants, token)
                if not self._clauses:
                    if allow_blank:
                        self._clauses = True
                    else:
                        raise ValueError("Blank specification not permitted")
                else:
                    self._optimize()

        # Check that any unconstrained variants actually do not vary
        if not isinstance(self._clauses, bool):
            bad_variants = set()
            for variant in valid_variants - constrained_variants:
                matches_true = any(
                    getattr(c, variant) and self.satisfies(c)
                    for c in self._concrete_cls.all_possible()
                )
                matches_false = any(
                    not getattr(c, variant) and self.satisfies(c)
                    for c in self._concrete_cls.all_possible()
                )
                if matches_true and matches_false:
                    bad_variants.add(variant)
            if bad_variants and strict:
                vs = ", ".join(sorted(bad_variants))
                raise ValueError(
                    f"Specification must constrain or explicitly mark unconstrained: {vs}"
                )

        # If required, check that we match *something*
        if not allow_empty:
            if isinstance(self._clauses, bool):
                matches_any = self._clauses
            else:
                matches_any = any(self.satisfies(c) for c in self._concrete_cls.all_possible())
            if not matches_any:
                raise ValueError(f"Specification does not match anything: {spec!r}")

    def satisfies(self, concrete: ConcreteSpecification) -> bool:
        """Test whether the given ConcreteSpecification satisfies this Specification."""
        if isinstance(self._clauses, bool):
            return self._clauses

        for vrs, c in self._clauses.items():
            cnt = sum(1 if getattr(concrete, v) else 0 for v in vrs)
            if cnt < c.min_enabled or c.max_enabled < cnt:
                return False
        return True

    def __str__(self) -> str:
        if isinstance(self._clauses, bool):
            return "all" if self._clauses else "none"

        fragments = []
        for vrs, cond in self._clauses.items():
            if len(vrs) == 1:
                assert cond.min_enabled == cond.max_enabled
                fragments.append(("+" if cond.min_enabled == 1 else "~") + vrs[0])
            else:
                cfrags = []
                if cond.min_enabled > 0:
                    cfrags.append(f"+>{cond.min_enabled:d}")
                if cond.max_enabled < len(vrs):
                    cfrags.append(f"+<{cond.max_enabled:d}")
                assert cfrags
                fragments.append(f"({' '.join(vrs)})[{' '.join(cfrags)}]")
        return " ".join(fragments)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({str(self)!r})"


class Configuration:
    """Representation of a possible build configuration of HPCToolkit."""

    args: list[str]

    def __init__(
        self,
        meson: Path,
        args: collections.abc.Iterable[str],
        comp: Compilers,
        variant: ConcreteSpecification,
    ):
        """Derive the full Configuration by adjusting the given args to meson setup to match
        the expected comp(ilers) and variant.
        """
        self.meson = meson
        self.manifest: Manifest = Manifest(mpi=variant.mpi, rocm=variant.rocm)

        self.args = [
            *args,
            f"-Dpapi={'enabled' if variant.papi else 'disabled'}",
            f"-Dpython={'enabled' if variant.python else 'disabled'}",
            f"-Dcuda={'enabled' if variant.cuda else 'disabled'}",
            f"-Dlevel0={'enabled' if variant.level0 else 'disabled'}",
            f"-Dgtpin={'enabled' if variant.gtpin else 'disabled'}",
            f"-Dopencl={'enabled' if variant.opencl else 'disabled'}",
            f"-Drocm={'enabled' if variant.rocm else 'disabled'}",
            f"-Dhpcprof_mpi={'enabled' if variant.mpi else 'disabled'}",
            f"-Dbuildtype={'debugoptimized' if variant.debug else 'release'}",
            f"-Dvalgrind_annotations={'true' if variant.valgrind_debug else 'false'}",
        ]

        self.native = MesonMachineFile()
        self.env: collections.abc.MutableMapping[str, str] = collections.ChainMap({}, os.environ)

        # Apply the configuration from the Compilers
        if comp.custom_compilers:
            cc, cxx = comp.custom_compilers
            self.native.add_binary("c", Path(cc))
            self.native.add_binary("cpp", Path(cxx))
            self.env["OMPI_CC"] = cc
            self.env["OMPI_CXX"] = cxx
