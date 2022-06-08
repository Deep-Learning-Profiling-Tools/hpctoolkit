#include "torch-monitor-api.h"

#include <stdlib.h>
#include <stdio.h>

#include <hpcrun/cct/cct.h>
#include <hpcrun/thread_data.h>
#include <hpcrun/metrics.h>

#include <torch_monitor.h>

#include "torch-monitor-logical.h"

#define TORCH_MONITOR_CALL(func, args)                     \
  do {                                                     \
    torch_monitor_status_t status = func args;             \
    if (status != TORCH_MONITOR_STATUS_SUCCESS) {          \
      fprintf(stderr, "Torch monitor status: %d", status); \
      exit(1);                                             \
    }                                                      \
  } while (0)

static bool torch_monitor_enabled = false;

bool
torch_monitor_status
(
 void
)
{
  return torch_monitor_enabled;
}


static void
torch_monitor_callback
(
 torch_monitor_callback_site_t callback_site,
 torch_monitor_callback_data_t* callback_data
)
{
  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    if (callback_data->domain == TORCH_MONITOR_DOMAIN_FUNCTION) {
      if (callback_data->data.op_data.sequence_number != -1) {
        // This op may have a corresponding backward call
        // TODO(Keren): record its forward call path
      }
    }
    // TODO(Keren)
    //if (callback_data->domain == TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION) {
    //}
    //if (callback_data->domain == TORCH_MONITOR_DOMAIN_MEMORY) {
    //}
  } else {
  }
}


void
torch_monitor_start
(
 bool native_stack
)
{
  torch_monitor_enabled = true;

  torch_monitor_logical_register(native_stack);

  TORCH_MONITOR_CALL(torch_monitor_domain_enable, (TORCH_MONITOR_DOMAIN_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_domain_enable, (TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION));
  TORCH_MONITOR_CALL(torch_monitor_callback_subscribe, (torch_monitor_callback));
  TORCH_MONITOR_CALL(torch_monitor_init, ());
}


void
torch_monitor_stop
(
 void
)
{
  torch_monitor_enabled = false;

  torch_monitor_logical_unregister();

  TORCH_MONITOR_CALL(torch_monitor_finalize, ());
}

