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
// Copyright ((c)) 2002-2020, Rice University
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

#ifndef torch_monitor_forward_cct_map_h
#define torch_monitor_forward_cct_map_h

#include <stdint.h>

#include <hpcrun/cct/cct.h>

typedef struct torch_monitor_forward_cct_map_entry_s torch_monitor_forward_cct_map_entry_t;

typedef struct forward_key_t {
  uint64_t forward_thread_id;
  int64_t sequence_number;
} forward_key_t;


void
torch_monitor_forward_cct_map_insert
(
 forward_key_t key,
 cct_node_t *cct
);


torch_monitor_forward_cct_map_entry_t *
torch_monitor_forward_cct_map_lookup
(
 forward_key_t key
);


void
torch_monitor_forward_cct_map_entry_cct_update
(
 torch_monitor_forward_cct_map_entry_t *entry,
 cct_node_t *cct
);


cct_node_t *
torch_monitor_forward_cct_map_entry_cct_get
(
 torch_monitor_forward_cct_map_entry_t *entry
);

#endif  // torch_monitor_forward_cct_map_h

