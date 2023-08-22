#ifndef TORCH_MONITOR_OP_PLACEHOLDERS_H
#define TORCH_MONITOR_OP_PLACEHOLDERS_H

//******************************************************************************
// system includes
//******************************************************************************

#include <stdint.h>

//******************************************************************************
// local includes
//******************************************************************************

#include <hpcrun/utilities/ip-normalized.h>

//******************************************************************************
// type declarations
//******************************************************************************

typedef enum torch_monitor_op_placeholder_type_t {
  torch_monitor_op_placeholder_type_forward =
      0,  // general copy, d2d d2a, or a2d
  torch_monitor_op_placeholder_type_backward = 1,
  torch_monitor_op_placeholder_type_count = 2
} torch_monitor_op_placeholder_type_t;

//******************************************************************************
// interface operations
//******************************************************************************

cct_node_t *torch_monitor_op_cct_insert(
    cct_node_t *api_node, torch_monitor_op_placeholder_type_t type);

ip_normalized_t torch_monitor_op_placeholder_ip(
    torch_monitor_op_placeholder_type_t type);

#endif
