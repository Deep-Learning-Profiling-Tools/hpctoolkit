// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2022, Rice University
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

//******************************************************************************
// system includes
//******************************************************************************

#include <assert.h>
#include <pthread.h>
#include <string.h>



//******************************************************************************
// local includes
//******************************************************************************

#include <hpcrun/cct/cct.h>

#include "lib/prof-lean/placeholders.h"
#include "torch-monitor-op-placeholders.h"

ip_normalized_t
torch_monitor_op_placeholder_ip
(
 torch_monitor_op_placeholder_type_t type
)
{
  switch(type) {
  #define CASE(N) case torch_monitor_op_placeholder_type_##N: return get_placeholder_norm(hpcrun_placeholder_torch_monitor_##N);
  CASE(forward)
  CASE(backward)
  #undef CASE
  case torch_monitor_op_placeholder_type_count:
    break;
  }
  assert(false && "Invalid torch_monitor placeholder type!");
  abort();
}

//******************************************************************************
// interface operations
//******************************************************************************

cct_node_t *
torch_monitor_op_cct_insert
(
 cct_node_t *api_node,
 torch_monitor_op_placeholder_type_t type
)
{
  return hpcrun_cct_insert_ip_norm(api_node, torch_monitor_op_placeholder_ip(type), true);
}
