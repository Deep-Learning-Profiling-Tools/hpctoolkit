# -*-Mode: makefile;-*-

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
## Copyright ((c)) 2002-2023, Rice University
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

AUTOMAKE_OPTIONS = foreign subdir-objects

include $(top_srcdir)/src/Makeinclude.config

#----------------------------------------------------------------------

BOOST_IFLAGS = @BOOST_IFLAGS@
DYNINST_IFLAGS = @DYNINST_IFLAGS@
LIBELF_INC = @LIBELF_INC@
TBB_IFLAGS = @TBB_IFLAGS@
XED2_INC = @XED2_INC@

if OPT_ENABLE_IGC
IGC_IFLAGS = @OPT_IGC_IFLAGS@
endif

if OPT_ENABLE_METRICS_DISCOVERY
MD_IFLAGS = @OPT_METRICS_DISCOVERY_IFLAGS@
endif

MYSOURCES = \
	gpu/GPUCFG.cpp  \
	gpu/GPUCFG_Cuda.cpp \
	gpu/GPUCFG_Intel.cpp \
	gpu/GPUCFGFactory.cpp  \
	gpu/GPUFunction.cpp  \
	gpu/GPUBlock.cpp  \
	gpu/GPUCodeSource.cpp  \
	gpu/GraphReader.cpp \
	gpu/CudaCFGParser.cpp  \
	Struct.cpp  \
	Struct-Inline.cpp  \
	Struct-Output.cpp

MYCXXFLAGS = \
	@HOST_CXXFLAGS@  \
	$(HPC_IFLAGS)  \
	$(BOOST_IFLAGS)  \
	$(DWARF_IFLAGS)  \
	-I$(LIBELF_INC)  \
	$(DYNINST_IFLAGS)  \
	$(IGC_IFLAGS) \
	$(MD_IFLAGS) \
	$(TBB_IFLAGS)

if HOST_CPU_X86_FAMILY
MYCXXFLAGS += -I$(XED2_INC)
endif

MYLIBADD = @HOST_LIBTREPOSITORY@

MYCLEAN = @HOST_LIBTREPOSITORY@

#----------------------------------------------------------------------

pkglibexecdir = @my_pkglibexecdir@

noinst_LTLIBRARIES = libHPCbanal.la

libHPCbanal_la_SOURCES  = $(MYSOURCES)
libHPCbanal_la_CXXFLAGS = $(MYCXXFLAGS)
libHPCbanal_la_LIBADD   = $(MYLIBADD)

if OPT_ENABLE_OPENMP
libHPCbanal_la_CXXFLAGS += $(OPENMP_FLAG)
endif

MOSTLYCLEANFILES = $(MYCLEAN)

#----------------------------------------------------------------------

include $(top_srcdir)/src/Makeinclude.rules
