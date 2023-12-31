project('hpctoolkit', 'c', 'cpp',
  meson_version: '>=1.1.0',
  license: 'BSD-3-Clause',
  default_options: ['buildtype=debugoptimized', 'c_std=gnu11', 'cpp_std=gnu++17', 'cuda_std=c++14'])

fs = import('fs')
pymod = import('python')

# Translate from prefixes to Meson dependency objects. They are missing most pieces like compiler
# args and such, but they are closer to the Mesonic.
foreach package : [
    'boost', 'bzip', 'dyninst', 'elfutils', 'tbb', 'libmonitor', 'libunwind', 'xerces', 'lzma',
    'zlib', 'libiberty', 'xed', 'memkind', 'yaml_cpp', 'papi', 'perfmon', 'opencl', 'gtpin',
    'igc', 'level0', 'rocm_hip', 'rocm_hsa', 'rocm_tracer', 'rocm_profiler', 'cuda',
]
    _p = meson.get_external_property(f'prefix_@package@', '')
    if _p != ''
        _p = fs.expanduser(_p)
        assert(fs.is_absolute(_p), f'property \'prefix_@package@\' must be an absolute path')
        assert(fs.is_dir(_p), f'property \'prefix_@package@\' must be a directory')
        _d = declare_dependency(variables: {'prefix': _p})
    else
        _d = dependency('', required: false)
    endif
    meson.override_dependency(f'external @package@', _d)
endforeach

# Collect all the dependencies we require for HPCToolkit
boost_exdep = dependency('external boost')
bzip_exdep = dependency('external bzip')
dyninst_exdep = dependency('external dyninst')
elfutils_exdep = dependency('external elfutils')
tbb_exdep = dependency('external tbb')
libmonitor_exdep = dependency('external libmonitor')
libunwind_exdep = dependency('external libunwind')
xerces_exdep = dependency('external xerces')
lzma_exdep = dependency('external lzma')
zlib_exdep = dependency('external zlib')
libiberty_exdep = dependency('external libiberty')
xed_exdep = dependency('external xed', required: host_machine.cpu_family() in ['x86', 'x86_64'])
memkind_exdep = dependency('external memkind')
yaml_cpp_exdep = dependency('external yaml_cpp')
papi_exdep = dependency('external papi', required: get_option('papi'))
if not papi_exdep.found()
    perfmon_exdep = dependency('external perfmon')
else
    perfmon_exdep = dependency('', required: false)
endif

# Find Python. This is a hard requirement since we use Python for the regression test suite as well.
python = pymod.find_installation('python3')

opencl_exdep = dependency('external opencl', required: get_option('opencl'))
level0_exdep = dependency('external level0', required: get_option('level0'))
_gtpin_f = get_option('gtpin').require(level0_exdep.found(),
    error_message: 'gtpin is only available with level0')
gtpin_exdep = dependency('external gtpin', required: _gtpin_f)
igc_exdep = dependency('external igc', required: _gtpin_f)

# Find CUDA language support.
cuda_exdep = dependency('external cuda', required: get_option('cuda'))
has_cuda = cuda_exdep.found()
if has_cuda
    add_languages('cuda', native: false, required: true)
    nvdisasm = find_program('nvdisasm', dirs: [cuda_exdep.get_variable(internal: 'prefix') / 'bin'])
endif

# Find HIP language support.
rocm_hip_exdep = dependency('external rocm_hip', required: get_option('rocm'))
rocm_hsa_exdep = dependency('external rocm_hsa', required: get_option('rocm'))
rocm_tracer_exdep = dependency('external rocm_tracer', required: get_option('rocm'))
rocm_profiler_exdep = dependency('external rocm_profiler', required: get_option('rocm'))
has_rocm = rocm_hip_exdep.found() and rocm_hsa_exdep.found() \
           and rocm_tracer_exdep.found() and rocm_profiler_exdep.found()
if has_rocm
    hipcc = find_program('hipcc', required: true)
    hipcc_cmd = [hipcc, '-o', '@OUTPUT@', '@INPUT@']
endif

# Find MPI. This is a stub since we really just pass it on to the Autogoo.
mpicxx = find_program('mpicxx', required: get_option('hpcprof_mpi'))

# See if we have runtime support for MPI, via the semi-standard mpiexec interface.
# Usage: [mpiexec, '<# of ranks>', command...]
mpiexec = find_program('mpiexec', required: get_option('hpcprof_mpi').disabled() ? get_option('hpcprof_mpi') : false)
if mpiexec.found()
    mpiexec_str = ';'.join(mpiexec.full_path(), '-host', 'localhost', '-n')
    mpiexec = [mpiexec, '-host', 'localhost', '-n']
else
    _sh = find_program('sh')
    mpiexec_str = ';'.join(_sh.full_path(), '-c', 'exit 77')
    mpiexec = [_sh, '-c', 'exit 77']
endif

# --- START AUTOGOO STUB ---------------------
# The build system is really Autotools, which is vastly different from Meson. We will eventually
# switch over from Autotools to Meson, but until we do this "stub" acts as the bridge between the
# two very different worlds. The goal is that currently written Meson code (i.e. regression tests)
# can remain as-is when the switch occurs, reducing some of the complexity that goes with this sort
# of thing.

# Regenerate the Autogoo during the Meson configuration phase. This technically edits the source
# tree, which is really *really* bad, but we don't have another option at this stage.
_make = find_program('make')
_autoenv = {
    'AUTORECONF': find_program('autoreconf', version: '=2.69').full_path(),
    'AUTOCONF': find_program('autoconf', version: '=2.69').full_path(),
    'ACLOCAL': find_program('aclocal', version: '=1.15.1').full_path(),
    'AUTOHEADER': find_program('autoheader', version: '=2.69').full_path(),
    'AUTOM4TE': find_program('autom4te', version: '=2.69').full_path(),
    'AUTOMAKE': find_program('automake', version: '=1.15.1').full_path(),
    'LIBTOOLIZE': find_program('libtoolize', version: '=2.4.6').full_path(),
    'M4': find_program('m4').full_path(),
    'MAKE': _make.full_path(),
}
message('Running ./autogen ...')
run_command(files('meson'/'autogen'),
    # The following files are read during the autogen process
    files('configure.ac'),
    files('doc'/'Makefile.am'),
    files('doc'/'man'/'Makefile.am'),
    files('Makefile.am'),
    files('src'/'Makefile.am'),
    files('src'/'extern'/'Makefile.am'),
    files('src'/'extern'/'libunwind'/'Makefile.am'),
    files('src'/'extern'/'lzma'/'Makefile.am'),
    files('src'/'lib'/'xml'/'Makefile.am'),
    files('src'/'lib'/'support-lean'/'Makefile.am'),
    files('src'/'lib'/'binutils'/'Makefile.am'),
    files('src'/'lib'/'Makefile.am'),
    files('src'/'lib'/'prof-lean'/'Makefile.am'),
    files('src'/'lib'/'analysis'/'Makefile.am'),
    files('src'/'lib'/'prof'/'Makefile.am'),
    files('src'/'lib'/'profile'/'Makefile.am'),
    files('src'/'lib'/'banal'/'Makefile.am'),
    files('src'/'lib'/'support'/'Makefile.am'),
    files('src'/'tool'/'hpcserver'/'Makefile.am'),
    files('src'/'tool'/'hpcserver'/'mpi'/'Makefile.am'),
    files('src'/'tool'/'hpcrun'/'utilities'/'bgq-cnk'/'Makefile.am'),
    files('src'/'tool'/'hpcrun'/'Makefile.am'),
    files('src'/'tool'/'Makefile.am'),
    files('src'/'tool'/'hpcfnbounds'/'Makefile.am'),
    files('src'/'tool'/'hpcstruct'/'Makefile.am'),
    files('src'/'tool'/'hpcproflm'/'Makefile.am'),
    files('src'/'tool'/'hpcproftt'/'Makefile.am'),
    files('src'/'tool'/'hpcprof-mpi'/'Makefile.am'),
    files('src'/'tool'/'hpctracedump'/'Makefile.am'),
    files('src'/'tool'/'hpcprof'/'Makefile.am'),
    files('lib'/'Makefile.am'),
    files('tests'/'Makefile.am'),
    # The following files are read by ./configure, but we put them here so that Meson regenerates
    # automatically when any of them change.
    # Passed to AC_CONFIG_FILES(...):
    files('doc/man/HPCToolkitVersionInfo.tex.in'),
    files('tests/Makefile.spack.in'),
    files('src/tool/hpcrun/scripts/hpcrun.in'),
    files('src/tool/hpcrun/scripts/hpclink.in'),
    files('src/tool/hpcrun/scripts/hpcsummary.in'),
    files('src/tool/hpcproftt/hpcproftt.in'),
    files('src/tool/hpcstruct/hpcstruct.in'),
    files('src/tool/hpcstruct/dotgraph.in'),
    files('tests/run-sort.in'),
    # Passed to AC_SUBST_FILE(...):
    files('config/copyright-hash'),
    files('config/export-hpctoolkit'),
    files('config/launch-options'),
    check: true, capture: false, env: _autoenv)

# Generate the full set of --with-* arguments for ./configure. This propagates the dependencies to
# the actual build.
_autogoo_args = [
    '--@0@-valgrind-annotations'.format(get_option('valgrind_annotations') ? 'enable' : 'disable'),
    '--with-boost=@0@'.format(boost_exdep.get_variable(internal: 'prefix')),
    '--with-bzip=@0@'.format(bzip_exdep.get_variable(internal: 'prefix')),
    '--with-dyninst=@0@'.format(dyninst_exdep.get_variable(internal: 'prefix')),
    '--with-elfutils=@0@'.format(elfutils_exdep.get_variable(internal: 'prefix')),
    '--with-tbb=@0@'.format(tbb_exdep.get_variable(internal: 'prefix')),
    '--with-libmonitor=@0@'.format(libmonitor_exdep.get_variable(internal: 'prefix')),
    '--with-libunwind=@0@'.format(libunwind_exdep.get_variable(internal: 'prefix')),
    '--with-xerces=@0@'.format(xerces_exdep.get_variable(internal: 'prefix')),
    '--with-lzma=@0@'.format(lzma_exdep.get_variable(internal: 'prefix')),
    '--with-zlib=@0@'.format(zlib_exdep.get_variable(internal: 'prefix')),
    '--with-libiberty=@0@'.format(libiberty_exdep.get_variable(internal: 'prefix')),
    '--with-memkind=@0@'.format(memkind_exdep.get_variable(internal: 'prefix')),
    '--with-yaml-cpp=@0@'.format(yaml_cpp_exdep.get_variable(internal: 'prefix')),
]
_autogoo_env = {}
if xed_exdep.found()
    _autogoo_args += ['--with-xed=@0@'.format(xed_exdep.get_variable(internal: 'prefix'))]
endif
if mpicxx.found()
    _autogoo_args += ['MPICXX=@0@'.format(mpicxx.full_path())]
endif
if papi_exdep.found()
    _autogoo_args += ['--with-papi=@0@'.format(papi_exdep.get_variable(internal: 'prefix'))]
endif
if perfmon_exdep.found()
    _autogoo_args += ['--with-perfmon=@0@'.format(perfmon_exdep.get_variable(internal: 'prefix'))]
endif
if get_option('python').enable_auto_if(python.found()).enabled()
    _autogoo_args += ['--with-python=@0@'.format(python.full_path() + '-config')]
endif
if opencl_exdep.found()
    _autogoo_args += ['--with-opencl=@0@'.format(opencl_exdep.get_variable(internal: 'prefix'))]
endif
if level0_exdep.found()
    _autogoo_args += ['--with-level0=@0@'.format(level0_exdep.get_variable(internal: 'prefix'))]
    if gtpin_exdep.found() and igc_exdep.found()
        _autogoo_args += [
            '--with-gtpin=@0@'.format(gtpin_exdep.get_variable(internal: 'prefix')),
            '--with-igc=@0@'.format(igc_exdep.get_variable(internal: 'prefix')),
        ]
    endif
endif
if has_rocm
    _autogoo_args += [
        '--with-rocm-hip=@0@'.format(rocm_hip_exdep.get_variable(internal: 'prefix')),
        '--with-rocm-hsa=@0@'.format(rocm_hsa_exdep.get_variable(internal: 'prefix')),
        '--with-rocm-tracer=@0@'.format(rocm_tracer_exdep.get_variable(internal: 'prefix')),
        '--with-rocm-profiler=@0@'.format(rocm_profiler_exdep.get_variable(internal: 'prefix')),
    ]
endif
if has_cuda
    _autogoo_args += ['--with-cuda=@0@'.format(cuda_exdep.get_variable(internal: 'prefix'))]
endif

# Set the compilers that should be used for compilations
_cc = meson.get_compiler('c')
assert(_cc.get_argument_syntax() == 'gcc', 'Compiler must support GCC-syntax arguments!')
_cpp = meson.get_compiler('cpp')
assert(_cpp.get_argument_syntax() == 'gcc', 'Compiler must support GCC-syntax arguments!')
_autogoo_args += [
    'CC=@0@'.format(' '.join(_cc.cmd_array())),
    'CXX=@0@'.format(' '.join(_cpp.cmd_array())),
]
if mpicxx.found()
    _autogoo_env += {
        'OMPI_CXX': ' '.join(_cpp.cmd_array()),
        'MPICH_CXX': ' '.join(_cpp.cmd_array()),
    }
endif

# Collect and translate the raw compiler flags
_c2flags = ['-Wstack-usage=65536']
if get_option('debug')
    _c2flags += ['-g']
endif
if get_option('optimization') != 'plain'
    _c2flags += ['-O' + get_option('optimization')]
endif
if get_option('warning_level') != '0'
    _warns = {
        '1': ['-Wall'],
        '2': ['-Wall', '-Wextra'],
        '3': ['-Wall', '-Wextra', '-Wpedantic'],
        'everything': ['-Weverything'],
    }
    _c2flags += _warns[get_option('warning_level')]
endif
if get_option('werror')
    _c2flags = ['-Werror']
endif
if get_option('b_ndebug') == 'true'
    _c2flags += ['-DNDEBUG']
endif

_cflags = [get_option('c_args')]
if get_option('c_std') != 'none'
    _cflags += ['-std=@0@'.format(get_option('c_std'))]
endif

_cppflags = [get_option('cpp_args')]
if get_option('cpp_std') != 'none'
    _cppflags += ['-std=@0@'.format(get_option('cpp_std'))]
endif

_autogoo_args += [
    '--enable-plain',
    'CFLAGS=@0@'.format(' '.join(_c2flags + _cflags)),
    'CXXFLAGS=@0@'.format(' '.join(_c2flags + _cppflags)),
]

# Translate any last built-in Meson options that have clean translations
_autogoo_args += [
    '--@0@-strip-debug'.format(get_option('strip') ? 'enable' : 'disable'),
    '--prefix=@0@'.format(get_option('prefix')),
]

# Run ./configure to configure the project as a whole
message('Running ./configure ...')
run_command(files('meson'/'configure'), 'autotools-build', files('configure'), _autogoo_args,
            check: true, capture: false, env: _autogoo_env)

# Build and install by deferring to a parallel make
_makepy = files('meson'/'make')
_autogoo_builddir = meson.project_build_root()/'autotools-build'
_makecmd = [_make, '-C', _autogoo_builddir, 'MAKE=@0@'.format(_make.full_path())]
run_target('makeclean', command: _makecmd + ['clean'], env: _autogoo_env)
custom_target('make -j', command: [python, _makepy] + _makecmd, console: true, env: _autogoo_env,
              build_always_stale: true, build_by_default: true, output: 'nothing.txt')
meson.add_install_script(python, _makepy, _make, '-C', _autogoo_builddir,
                         'MAKE=@0@'.format(_make.full_path()), 'install')

# Once HPCToolkit is installed, they'll be in these paths
hpcrun = get_option('prefix') / 'bin' / 'hpcrun'
hpcstruct = get_option('prefix') / 'bin' / 'hpcstruct'
hpcprof = get_option('prefix') / 'bin' / 'hpcprof'
if get_option('hpcprof_mpi').enable_auto_if(mpicxx.found()).enabled()
  hpcprof_mpi = get_option('prefix') / 'bin' / 'hpcprof-mpi'
endif
# --- END AUTOGOO STUB -----------------------

# Regression tests are in the tests2 subdir, for now.
subdir('tests2')
