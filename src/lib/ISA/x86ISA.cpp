// $Id$
// -*-C++-*-
// * BeginRiceCopyright *****************************************************
// 
// Copyright ((c)) 2002, Rice University 
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
// 
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
// 
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage. 
// 
// ******************************************************* EndRiceCopyright *

//***************************************************************************
//
// File:
//    i686ISA.C
//
// Purpose:
//    [The purpose of this file]
//
// Description:
//    [The set of functions, macros, etc. defined in the file]
//
//***************************************************************************

//************************* System Include Files ****************************

#ifdef NO_STD_CHEADERS
# include <stdarg.h>
# include <string.h>
#else
# include <cstdarg>
# include <cstring> // for 'memcpy'
using namespace std; // For compatibility with non-std C headers
#endif

//*************************** User Include Files ****************************

#include "i686ISA.h"
#include <include/gnu_dis-asm.h>

//*************************** Forward Declarations ***************************

//****************************************************************************

extern "C" { 

static int fake_fprintf_func(PTR, const char *fmt, ...)
{
  //va_list ap;
  //va_start(ap, fmt);
  return 0;
}

static void print_addr (bfd_vma vma, struct disassemble_info *info)
{
  info->fprintf_func (info->stream, "0x%08x", vma);
}

static int read_memory_func (bfd_vma vma, bfd_byte *myaddr, unsigned int len,
			     struct disassemble_info *info)
{
  memcpy(myaddr, BFDVMA_TO_PTR(vma, const char*), len);
  return 0; /* success */
}

} // extern "C"

//****************************************************************************
// i686ISA
//****************************************************************************

i686ISA::i686ISA()
{
  // See 'dis-asm.h'
  di = new disassemble_info;
  INIT_DISASSEMBLE_INFO(*di, stdout, fake_fprintf_func);
  //  fprintf_func: (int (*)(void *, const char *, ...))fprintf;

  di->arch = bfd_arch_i386;      // bfd_get_arch(abfd);
  di->mach = bfd_mach_i386_i386; // bfd_get_mach(abfd); needed in print_insn()
  di->endian = BFD_ENDIAN_LITTLE;
  di->read_memory_func = read_memory_func; // vs. 'buffer_read_memory'
  di->print_address_func = print_addr;     // vs. 'generic_print_address'
}

i686ISA::~i686ISA()
{
  delete di;
}

ushort
i686ISA::GetInstSize(MachInst* mi)
{
  ushort size;
  DecodingCache *cache;

  if ((cache = CacheLookup(mi)) == NULL) {
    size = print_insn_i386(PTR_TO_BFDVMA(mi), di);
    CacheSet(mi, size);
  } else {
    size = cache->instSize;
  }
  return size;
}

ISA::InstDesc
i686ISA::GetInstDesc(MachInst* mi, ushort opIndex, ushort s)
{
  ISA::InstDesc d;

  if (CacheLookup(mi) == NULL) {
    ushort size = print_insn_i386(PTR_TO_BFDVMA(mi), di);
    CacheSet(mi, size);
  }

  switch(di->insn_type) {
    case dis_noninsn:
      d.Set(InstDesc::INVALID);
      break;
    case dis_branch:
      if (di->target != 0) {
	d.Set(InstDesc::BR_UN_COND_REL);
      } else {
	d.Set(InstDesc::BR_UN_COND_IND);
      }
      break;
    case dis_condbranch:
      if (di->target != 0) {
	d.Set(InstDesc::INT_BR_COND_REL); // arbitrarily choose int
      } else {
	d.Set(InstDesc::INT_BR_COND_IND); // arbitrarily choose int
      }
      break;
    case dis_jsr:
      if (di->target != 0) {
	d.Set(InstDesc::SUBR_REL);
      } else {
	d.Set(InstDesc::SUBR_IND);
      }
      break;
    case dis_condjsr:
      d.Set(InstDesc::OTHER);
    case dis_dref:
    case dis_dref2:
      d.Set(InstDesc::MEM_OTHER);
    default:
      d.Set(InstDesc::OTHER);
      break;
  }
  return d;
}

Addr
i686ISA::GetInstTargetAddr(MachInst* mi, Addr pc, ushort opIndex, ushort sz)
{
  if (CacheLookup(mi) == NULL) {
    ushort size = print_insn_i386(PTR_TO_BFDVMA(mi), di);
    CacheSet(mi, size);
  }

  // N.B.: The GNU decoders expect that the address of the 'mi' is
  // actually the PC/vma.  Furthermore, because i686 is 32-bit
  // decoding, it only keeps 32 bits of 'mi'.

  // The target field is only set on instructions with targets.
  if (di->target != 0) {
    return (di->target - (PTR_TO_BFDVMA(mi) & 0xffffffff)) + (bfd_vma)pc;
  } else {
    return 0;
  }
}


