#ifndef torch_monitor_logical_h
#define torch_monitor_logical_h

#include <stdbool.h>

#include <hpcrun/cct/cct.h>
#include <hpcrun/metrics.h>
#include <hpcrun/thread_data.h>

bool torch_monitor_native_stack_status();

void
torch_monitor_logical_register
(
 bool native_stack
);

void
torch_monitor_logical_unregister
(
 void
);

// Active backtrace: python stack only
cct_node_t *
torch_monitor_backtrace2cct
(
 cct_bundle_t *cct,
 int metric_id,
 hpcrun_metricVal_t metric_incr
);

#endif  // torch_monitor_logical_h
