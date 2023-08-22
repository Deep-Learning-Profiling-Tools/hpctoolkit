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

ip_normalized_t torch_monitor_op_placeholder_ip(
    torch_monitor_op_placeholder_type_t type) {
  switch (type) {
#define CASE(N)                               \
  case torch_monitor_op_placeholder_type_##N: \
    return get_placeholder_norm(hpcrun_placeholder_torch_monitor_##N);
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

cct_node_t *torch_monitor_op_cct_insert(
    cct_node_t *api_node, torch_monitor_op_placeholder_type_t type) {
  return hpcrun_cct_insert_ip_norm(api_node,
                                   torch_monitor_op_placeholder_ip(type), true);
}
