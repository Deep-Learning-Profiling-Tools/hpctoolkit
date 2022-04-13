// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// $HeadURL: https://outreach.scidac.gov/svn/hpctoolkit/trunk/src/tool/hpcrun/sample-sources/papi.c $
// $Id: papi.c 3328 2010-1/2-23 23:39:09Z tallent $
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

//
// Monitoring pytorch APIs using libtorch_monitor
//

//******************************************************************************
// system includes
//******************************************************************************

#ifndef HPCRUN_STATIC_LINK
#include <dlfcn.h>
#endif

#define TORCH_MONITOR_STR "torch_monitor"

//******************************************************************************
// local includes
//******************************************************************************

#include "../libdl.h"
#include "../simple_oo.h"
#include "../sample_source_obj.h"
#include "../common.h"

#include <hpcrun/device-finalizers.h>
#include <hpcrun/messages/messages.h>
#include <hpcrun/control-knob.h>
#include <hpcrun/thread_data.h>

#include "torch-monitor-api.h"

static device_finalizer_fn_entry_t device_finalizer_shutdown;

//******************************************************************************
// interface operations
//******************************************************************************

static void
METHOD_FN(init)
{
  self->state = INIT;

  control_knob_register("HPCRUN_TORCH_MONITOR_NATIVE_STACK_ENABLE", "FALSE", ck_string);
}

static void
METHOD_FN(thread_init)
{
  TMSG(TORCH_MONITOR, "thread_init");
}

static void
METHOD_FN(thread_init_action)
{
  TMSG(TORCH_MONITOR, "thread_init_action");
}
static void
METHOD_FN(start)
{
  TMSG(TORCH_MONITOR, "start");
  TD_GET(ss_state)[self->sel_idx] = START;
}

static void
METHOD_FN(thread_fini_action)
{
  TMSG(TORCH_MONITOR, "thread_fini_action");
}

static void
METHOD_FN(stop)
{
  TD_GET(ss_state)[self->sel_idx] = STOP;
}

static void
METHOD_FN(shutdown)
{
  self->state = UNINIT;
}


static bool
METHOD_FN(supports_event, const char *ev_str)
{
#ifndef HPCRUN_STATIC_LINK
  return hpcrun_ev_is(ev_str, TORCH_MONITOR_STR);
#else
  return false;
#endif
}

static void
METHOD_FN(process_event_list, int lush_metrics)
{
  bool native_stack = false;
  char *native_stack_enable_str = NULL;
  control_knob_value_get_string("HPCRUN_TORCH_MONITOR_NATIVE_STACK_ENABLE", &native_stack_enable_str);
  if (strcmp(native_stack_enable_str, "TRUE") == 0) {
    native_stack = true;
  }

  torch_monitor_start(native_stack);

  device_finalizer_shutdown.fn = torch_monitor_stop;
  device_finalizer_register(device_finalizer_type_shutdown, &device_finalizer_shutdown);
}

static void
METHOD_FN(finalize_event_list)
{
}

static void
METHOD_FN(gen_event_set,int lush_metrics)
{
}

static void
METHOD_FN(display_events)
{
  printf("===========================================================================\n");
  printf("Tracing with torch-monitor\n");
  printf("===========================================================================\n");
}


//******************************************************************************
// object
//******************************************************************************

#define ss_name torch_monitor
#define ss_cls SS_SOFTWARE

#include "../ss_obj.h"
