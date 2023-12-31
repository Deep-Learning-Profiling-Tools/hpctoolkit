import collections
import contextlib
import hashlib
import json
import linecache
import os
import re
import shlex
import shutil
import subprocess
import tempfile
import typing
from pathlib import Path

from .action import Action, ActionResult, ReturnCodeResult
from .configuration import Configuration
from .logs import dump_file
from .util import nproc, nproc_max, project_dir


@contextlib.contextmanager
def _stdlogfiles(logdir: Path | None, logprefix: str, suffix1: str, suffix2: str | None = None):
    if logdir is not None:
        logdir = Path(logdir)
        with open(logdir / (logprefix + suffix1), "w", encoding="utf-8") as f1:
            if suffix2 is not None:
                with open(logdir / (logprefix + suffix2), "w", encoding="utf-8") as f2:
                    yield f1, f2
            else:
                yield f1, f1
    else:
        yield None, None


class Setup(Action):
    """Setup a build directory based on a Configuration."""

    def __init__(self):
        self._compiler: tuple[str, str] | None = None

    def name(self) -> str:
        return "meson setup"

    def dependencies(self) -> tuple[Action, ...]:
        return ()

    def run(
        self,
        cfg: Configuration,
        *,
        builddir: Path,
        srcdir: Path,
        installdir: Path,
        logdir: Path | None = None,
    ) -> ActionResult:
        srcdir.resolve(strict=True)
        builddir.mkdir()
        cfg_path = builddir / "cfg.ini"
        cfg.native.save(cfg_path)

        cmd: list[str | Path] = [cfg.meson, "setup"]
        cmd.extend(cfg.args)
        cmd.append(f"--prefix={installdir.as_posix()}")
        cmd.append(f"--native-file={cfg_path}")
        cmd.append(builddir)
        cmd.append(srcdir)
        with _stdlogfiles(logdir, "configure", ".log") as (config_log, _):
            proc = subprocess.run(
                cmd, stdout=config_log, stderr=config_log, env=cfg.env, check=False
            )
        if logdir is not None:
            shutil.copyfile(
                builddir / "meson-logs" / "meson-log.txt", logdir / "configure.meson.log"
            )
            if (builddir / "autotools-build" / "config.log").exists():
                shutil.copyfile(
                    builddir / "autotools-build" / "config.log", logdir / "configure.config.log"
                )

        if proc.returncode != 0 and logdir is not None:
            dump_file(logdir / "configure.log")

        return ReturnCodeResult("configure", proc.returncode)


class MesonAction(Action):
    """Base class for Actions that primarily run `meson ...` in the build directory."""

    def _run(
        self,
        logprefix: str,
        cfg: Configuration,
        builddir: Path,
        *targets,
        env: dict[str, str] | None = None,
        logdir: Path | None = None,
        split_stderr: bool = True,
    ) -> ActionResult:
        # pylint: disable=too-many-locals
        assert builddir.is_dir()

        final_env = cfg.env
        if env is not None:
            final_env = collections.ChainMap(env, final_env)

        cmd = [cfg.meson, *targets]

        logsuffixes = (".stdout.log", ".stderr.log") if split_stderr else (".log",)
        with _stdlogfiles(logdir, logprefix, *logsuffixes) as (out_log, err_log):
            proc = subprocess.run(
                cmd, cwd=builddir, stdout=out_log, stderr=err_log, env=final_env, check=False
            )

        if logdir is not None:
            dump_file(logdir / (logprefix + logsuffixes[-1]))

        return ReturnCodeResult(" ".join(["meson"] + [str(t) for t in targets]), proc.returncode)


class BuildResult(ActionResult):
    """Detect warnings/errors in build logs."""

    def _gcc_to_cq(self, logline: str) -> dict | None:
        # Compiler warning regex:
        #     {path}.{extension}:{line}:{column}: {severity}: {message} [{flag(s)}]
        mat = re.fullmatch(
            r"^(.*)\.([a-z+]{1,3}):(\d+):(\d+:)?\s+(warning|error):\s+(.*)\s+\[((\w|-|=)*)\]$",
            logline.strip("\n"),
        )
        if not mat:
            return None

        report: dict[str, typing.Any] = {
            "type": "issue",
            "check_name": mat.group(7),  # flag(2)
            "description": mat.group(6),  # message
            "categories": ["compiler"],
            "location": {},
        }

        topdir = project_dir()
        path = Path(mat.group(1) + "." + mat.group(2))
        if path.is_absolute() and not path.is_relative_to(topdir):
            return None
        if not path.is_absolute():
            # Strip off prefixes and see if we can find the file we're looking for
            for i in range(1, len(path.parts) - 1):
                newpath = topdir / Path(*path.parts[i:])
                if newpath.is_file():
                    path = newpath
                    break
            else:
                return None
        report["location"]["path"] = path.relative_to(topdir).as_posix()

        line = int(mat.group(3))
        if len(mat.group(4)) > 0:
            col = int(mat.group(4)[:-1])
            report["location"]["positions"] = {"begin": {"line": line, "column": col}}
        else:
            report["location"]["lines"] = {"begin": line}

        match mat.group(5):
            case "warning":
                report["severity"] = "major"
                self.warnings += 1
            case "error":
                report["severity"] = "critical"
                self.errors += 1
        assert "severity" in report

        # The fingerprint is the hash of the report in JSON form, with parts masked out
        report["fingerprint"] = hashlib.md5(
            json.dumps(
                report
                | {
                    "location": report["location"]
                    | {"positions": None, "lines": None, "line": linecache.getline(str(path), line)}
                }
            ).encode("utf-8")
        ).hexdigest()
        return report

    def __init__(self, logfile: Path, cq_output: Path | None, subresult: ActionResult):
        self.subresult = subresult
        self.warnings, self.errors = 0, 0
        report = []
        with open(logfile, encoding="utf-8") as f:
            for line in f:
                cq = self._gcc_to_cq(line)
                if cq is not None:
                    report.append(cq)
                elif re.match(r"[^:]+:(\d+:){1,2}\s+warning:", line):  # Warning from GCC
                    self.warnings += 1
                elif re.match(r"[^:]+:(\d+:){1,2}\s+error:", line) or re.match(
                    r"[^:]+:\s+undefined reference to", line
                ):  # Error from GCC or ld
                    self.errors += 1
        if cq_output is not None:
            with open(cq_output, "w", encoding="utf-8") as f:
                json.dump(report, f)

    @property
    def completed(self):
        return self.subresult.completed

    @property
    def passed(self):
        return self.subresult.passed and self.errors == 0

    @property
    def flawless(self):
        return self.subresult.flawless and self.errors == 0 and self.warnings == 0

    def summary(self):
        fragments = []
        if self.errors > 0:
            fragments.append(
                f"detected {self.errors:d} errors + {self.warnings:d} warnings during build"
            )
        elif self.warnings > 0:
            fragments.append(f"detected {self.warnings:d} warnings during build")

        if not self.subresult.flawless:
            fragments.append(self.subresult.summary())

        return ", ".join(fragments) if fragments else self.subresult.summary()


class UnscannedBuildResult(ActionResult):
    """Result used when the build logs were not saved and thus not scanned."""

    def __init__(self, subresult: ActionResult):
        self.subresult = subresult

    @property
    def completed(self):
        return self.subresult.completed

    @property
    def passed(self):
        return False

    @property
    def flawless(self):
        return False

    def summary(self):
        return (
            "build logs not saved, error detection skipped"
            if self.subresult.flawless
            else self.subresult.summary()
        )


class Build(MesonAction):
    """Build the configured build directory."""

    def name(self) -> str:
        return "meson compile"

    def dependencies(self) -> tuple[Action]:
        return (Setup(),)

    def run(
        self,
        cfg: Configuration,
        *,
        builddir: Path,
        srcdir: Path,
        installdir: Path,
        logdir: Path | None = None,
    ) -> ActionResult:
        res = self._run(
            "build", cfg, builddir, "compile", f"-j{nproc():d}", logdir=logdir, split_stderr=True
        )
        return (
            BuildResult(logdir / "build.stderr.log", logdir / "build.cq.json", res)
            if logdir is not None
            else UnscannedBuildResult(res)
        )


class Install(MesonAction):
    """Install the configured build directory."""

    def name(self) -> str:
        return "meson install"

    def dependencies(self) -> tuple[Action]:
        return (Build(),)

    def run(
        self,
        cfg: Configuration,
        *,
        builddir: Path,
        srcdir: Path,
        installdir: Path,
        logdir: Path | None = None,
    ) -> ActionResult:
        return self._run("install", cfg, builddir, "install", logdir=logdir, split_stderr=True)


class CheckManifestResult(ActionResult):
    """Result from checking against the expected install manifest."""

    def __init__(self, missing: int, unexpected: int):
        self.missing = missing
        self.unexpected = unexpected

    @property
    def completed(self):
        return True

    @property
    def passed(self):
        return self.missing == 0

    @property
    def flawless(self):
        return self.passed and self.unexpected == 0

    def summary(self):
        if self.missing > 0:
            return f"{self.missing:d} files not installed + {self.unexpected:d} unexpected installed files"
        if self.unexpected > 0:
            return f"{self.unexpected:d} unexpected installed files"
        return "install manifest matched"


class CheckInstallManifest(Action):
    """Check the installed files against the expected install manifest."""

    def header(self, cfg: Configuration) -> str:
        return "Checking install manifest"

    def name(self) -> str:
        return "check install"

    def dependencies(self) -> tuple[Action]:
        return (Install(),)

    def run(
        self,
        cfg: Configuration,
        *,
        builddir: Path,
        srcdir: Path,
        installdir: Path,
        logdir: Path | None = None,
    ) -> ActionResult:
        return CheckManifestResult(*cfg.manifest.check(installdir))


class MissingJUnitLogs(ActionResult):
    """Result returned when the expected JUnit logs weren't generated by Test."""

    def __init__(self, fn: str, subresult):
        self.subresult = subresult
        self.filename = fn

    @property
    def completed(self):
        return False

    @property
    def passed(self):
        return False

    @property
    def flawless(self):
        return False

    def summary(self):
        frag = f"JUnit log {self.filename} missing"
        return frag if self.subresult.flawless else f"{self.subresult.summary()}, {frag}"


class Test(MesonAction):
    """Run build-time tests for the configured build directory."""

    def __init__(self):
        self.junit_copyout = False
        self.meson_opts = []

    def name(self) -> str:
        return f"meson test {shlex.join(self.meson_opts)}"

    def dependencies(self) -> tuple[Action, ...]:
        return Build(), Install()

    def run(
        self,
        cfg: Configuration,
        *,
        builddir: Path,
        srcdir: Path,
        installdir: Path,
        logdir: Path | None = None,
    ) -> ActionResult:
        # pylint: disable=too-many-locals
        res = self._run(
            "test",
            cfg,
            builddir,
            "test",
            *self.meson_opts,
            logdir=logdir,
            env={"MESON_TESTTHREADS": f"{nproc_max():d}"},
            split_stderr=False,
        )

        if logdir is not None:
            testlogdir = logdir / "test"
            testlogdir.mkdir()
            shutil.copyfile(builddir / "meson-logs" / "testlog.txt", testlogdir / "test.log")

        junit_log = builddir / "meson-logs" / "testlog.junit.xml"
        if not junit_log.exists():
            return MissingJUnitLogs(junit_log.name, res)

        if logdir is not None:
            shutil.copyfile(junit_log, logdir / "test.junit.xml")

        if self.junit_copyout:
            with open(junit_log, "rb") as inf, os.fdopen(
                tempfile.mkstemp(prefix="test.", suffix=".junit.xml", dir=os.getcwd())[0], "wb"
            ) as outf:
                shutil.copyfileobj(inf, outf)

        return res


class UnsavedTestDataResult(ActionResult):
    """Result used when the generated test data was not saved."""

    def __init__(self, subresult: ActionResult):
        self.subresult = subresult

    @property
    def completed(self):
        return self.subresult.completed

    @property
    def passed(self):
        return self.subresult.passed

    @property
    def flawless(self):
        return False

    def summary(self):
        return "test data was not saved" if self.subresult.flawless else self.subresult.summary()


class FreshTestData(MesonAction):
    """Generate data for later, offline tests."""

    def name(self) -> str:
        return f"meson compile {self.tarball_path}"

    def dependencies(self) -> tuple[Action, ...]:
        return Build(), Install()

    @property
    def suite(self) -> str:
        raise NotImplementedError

    suites: typing.ClassVar[dict[str, type["FreshTestData"]]] = {}

    @classmethod
    def register(cls, subcls):
        cls.suites[subcls.suite] = subcls
        return subcls

    @property
    def tarball_path(self) -> Path:
        return Path("tests2") / "data" / f"fresh-testdata-{self.suite}.tar.xz"

    def run(
        self,
        cfg: Configuration,
        *,
        builddir: Path,
        srcdir: Path,
        installdir: Path,
        logdir: Path | None = None,
    ) -> ActionResult:
        result = self._run(
            f"fresh-testdata-{self.suite}",
            cfg,
            builddir,
            "compile",
            f"-j{nproc():d}",
            self.tarball_path,
            logdir=logdir,
        )
        if logdir is not None:
            if result.completed:
                shutil.copyfile(
                    builddir / self.tarball_path,
                    logdir / f"fresh-testdata-{self.suite}.tar.xz",
                )
            return result
        return UnsavedTestDataResult(result)


@FreshTestData.register
class FreshTestDataNone(FreshTestData):
    suite: typing.ClassVar[str] = "none"


@FreshTestData.register
class FreshTestDataCPU(FreshTestData):
    suite: typing.ClassVar[str] = "cpu"


@FreshTestData.register
class FreshTestDataNvidia(FreshTestData):
    suite: typing.ClassVar[str] = "nvidia"


@FreshTestData.register
class FreshTestDataAMD(FreshTestData):
    suite: typing.ClassVar[str] = "amd"


@FreshTestData.register
class FreshTestDataSWCuda(FreshTestData):
    suite: typing.ClassVar[str] = "sw-cuda"


@FreshTestData.register
class FreshTestDataX8664(FreshTestData):
    suite: typing.ClassVar[str] = "x86-64"
