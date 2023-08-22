#ifndef TORCH_MONITOR_FUNCTION_CCT_MAP_H
#define TORCH_MONITOR_FUNCTION_CCT_MAP_H

#include <hpcrun/cct/cct.h>
#include <stdint.h>

typedef struct torch_monitor_function_cct_map_entry_s
    torch_monitor_function_cct_map_entry_t;

typedef struct function_key_t {
  uint64_t forward_thread_id;
  int64_t sequence_number;
} function_key_t;

void torch_monitor_function_cct_map_insert(function_key_t key, cct_node_t *cct);

torch_monitor_function_cct_map_entry_t *torch_monitor_function_cct_map_lookup(
    function_key_t key);

void torch_monitor_function_cct_map_entry_cct_update(
    torch_monitor_function_cct_map_entry_t *entry, cct_node_t *cct);

cct_node_t *torch_monitor_function_cct_map_entry_cct_get(
    torch_monitor_function_cct_map_entry_t *entry);

#endif  // TORCH_MONITOR_FUNCTION_CCT_MAP_H
