#!/usr/bin/env python3

import sys

from hpctoolkit.formats import from_path  # noqa: import-error
from hpctoolkit.formats.diff.strict import StrictAccuracy, StrictDiff  # noqa: import-error
from hpctoolkit.test.execution import hpcprof_mpi, thread_disruptive  # noqa: import-error
from hpctoolkit.test.tarball import extracted  # noqa: import-error

rankcnt = int(sys.argv[1])
threadcnt = int(sys.argv[2])

with extracted(sys.argv[4]) as dbase:
    base = from_path(dbase)

with extracted(sys.argv[3]) as meas:
    with thread_disruptive():
        with hpcprof_mpi(rankcnt, meas, f"-j{threadcnt:d}", "--foreign") as db:
            diff = StrictDiff(base, from_path(db.basedir))
            acc = StrictAccuracy(diff)
            if len(diff.hunks) > 0 or acc.inaccuracy:
                diff.render(sys.stdout)
                acc.render(sys.stdout)
                sys.exit(1)
