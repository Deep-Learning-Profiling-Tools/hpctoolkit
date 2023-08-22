#ifndef TORCH_MONITOR_LOGICAL_H
#define TORCH_MONITOR_LOGICAL_H

#include <hpcrun/cct/cct.h>
#include <hpcrun/messages/messages.h>
#include <hpcrun/metrics.h>
#include <hpcrun/thread_data.h>
#include <stdbool.h>

#define TORCH_MONITOR_MSG(...)         \
  do {                                 \
    bool unsafe = hpcrun_safe_enter(); \
    TMSG(TORCH_MONITOR, __VA_ARGS__);  \
    if (unsafe) hpcrun_safe_exit();    \
  } while (0)

bool torch_monitor_native_stack_status_get();

int torch_monitor_python_module_id_get();

void torch_monitor_logical_register(bool native_stack);

void torch_monitor_logical_unregister(void);

ip_normalized_t torch_monitor_function_ip(const char *function_name);

// Active backtrace: python stack only
cct_node_t *torch_monitor_backtrace2cct(cct_bundle_t *cct, int metric_id,
                                        hpcrun_metricVal_t metric_incr);

#endif  // TORCH_MONITOR_LOGICAL_H
