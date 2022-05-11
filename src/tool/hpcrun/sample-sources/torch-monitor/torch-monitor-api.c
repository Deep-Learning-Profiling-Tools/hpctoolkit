#include "torch-monitor-api.h"

#include <stdlib.h>
#include <stdio.h>

#include <hpcrun/cct/cct.h>
#include <hpcrun/thread_data.h>
#include <hpcrun/metrics.h>
#include <hpcrun/sample_event.h>
#include <hpcrun/safe-sampling.h>

#include <torch_monitor.h>

#include "torch-monitor-logical.h"
#include "torch-monitor-thread-obj.h"
#include "torch-monitor-op-placeholders.h"
#include "torch-monitor-forward-cct-map.h"

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


static cct_node_t *
forward_cct_get
(
 const char *function_name,
 torch_monitor_thread_obj_t *thread_obj
)
{
  torch_monitor_python_state_get(thread_obj->python_max_num_states, thread_obj->python_states,
    &thread_obj->python_cur_num_states);

  hpcrun_metricVal_t zero_metric_incr = {.i = 0};
  int zero_metric_id = 0;  // nothing to see here

  hpcrun_safe_enter(); 

  ucontext_t uc;
  getcontext(&uc); // current context, where unwind will begin 

  cct_node_t *cct = hpcrun_sample_callpath(&uc, zero_metric_id, zero_metric_incr, 0, 1, NULL).sample_node;
  if (function_name != NULL) {
    cct = torch_monitor_backtrace_function_insert(cct, function_name);
  }

  hpcrun_safe_exit();

  return cct;
}


static void
forward_prev_cct_update
(
 cct_node_t *cct,
 torch_monitor_thread_obj_t *thread_obj
)
{
  thread_obj->prev_cct = torch_monitor_op_cct_insert(cct, torch_monitor_op_placeholder_type_forward);
}


static void
forward_function_callback
(
 const torch_monitor_callback_data_t *callback_data,
 torch_monitor_thread_obj_t *thread_obj
)
{
  uint64_t forward_thread_id = callback_data->current_thread_id;
  int64_t sequence_number = callback_data->data.op_data.sequence_number;
  uint32_t nested_level = callback_data->data.op_data.nested_level;
  const char *function_name = callback_data->data.op_data.name;
 
  thread_obj->thread_state |= TORCH_MONITOR_THREAD_STATE_FORWARD;

  TMSG(TORCH_MONITOR, "Enter forward level %u state %p", nested_level, thread_obj->thread_state);

  if (sequence_number == -1 || nested_level != 0) {
    // sequence_number == -1: This op may not have a corresponding backward call
    // nested_level != 0: This op isn't the entry to aten
    return;
  }

  cct_node_t *cct = forward_cct_get(function_name, thread_obj);
  forward_prev_cct_update(cct, thread_obj);

  // A node in a computation graph can be without backward operations.
  // In this case, the <forward_thread_id, sequence_number> pair could repeat,
  // so that we could have multiple pair->cct mappings.
  // We memoize the last pair->cct mapping and use the cct to relate the correponsding backward op.
  forward_key_t key = {
    .forward_thread_id = forward_thread_id,
    .sequence_number = sequence_number
  };
  torch_monitor_forward_cct_map_entry_t *entry = torch_monitor_forward_cct_map_lookup(key);
  if (entry == NULL) {
    TMSG(TORCH_MONITOR, "Insert forward_thread_id %lu sequence_number %ld", forward_thread_id, sequence_number);
    torch_monitor_forward_cct_map_insert(key, cct);
  } else {
    TMSG(TORCH_MONITOR, "Update forward_thread_id %lu sequence_number %ld", forward_thread_id, sequence_number);
    // We can update without holding a lock.
    // When a forward op is in progress, its backward op has not started
    torch_monitor_forward_cct_map_entry_cct_update(entry, cct);
  }
}


static void
backward_prev_cct_update
(
 cct_node_t *cct,
 torch_monitor_thread_obj_t *thread_obj
)
{
  thread_data_t* td = hpcrun_get_thread_data();
  cct = hpcrun_cct_insert_path_return_leaf(td->core_profile_trace_data.epoch->csdata.tree_root, cct);
  cct = torch_monitor_op_cct_insert(cct, torch_monitor_op_placeholder_type_backward);
  thread_obj->prev_cct = cct;
}


static void
backward_function_callback
(
 const torch_monitor_callback_data_t *callback_data,
 torch_monitor_thread_obj_t *thread_obj
)
{
  uint64_t forward_thread_id = callback_data->data.op_data.forward_thread_id;
  int64_t sequence_number = callback_data->data.op_data.sequence_number;
  uint32_t nested_level = callback_data->data.op_data.nested_level;

  thread_obj->thread_state |= TORCH_MONITOR_THREAD_STATE_BACKWARD;

  TMSG(TORCH_MONITOR, "Enter backward level %u state %p", nested_level, thread_obj->thread_state);

  if (sequence_number == -1) {
    // This op may not have a corresponding backward call
    return;
  }

  forward_key_t key = {
    .forward_thread_id = forward_thread_id,
    .sequence_number = sequence_number
  };

  torch_monitor_forward_cct_map_entry_t *entry = torch_monitor_forward_cct_map_lookup(key);
  if (entry != NULL) {
    TMSG(TORCH_MONITOR, "Lookup forward_thread_id %lu sequence_number %ld", forward_thread_id, sequence_number);
    cct_node_t *cct = torch_monitor_forward_cct_map_entry_cct_get(entry);
    assert(cct != NULL);
    backward_prev_cct_update(cct, thread_obj);
  }
}


static void
forward_cleanup_callback
(
 const torch_monitor_callback_data_t *callback_data,
 torch_monitor_thread_obj_t *thread_obj
)
{
  uint32_t nested_level = callback_data->data.op_data.nested_level;

  if ((thread_obj->thread_state & TORCH_MONITOR_THREAD_STATE_BACKWARD) == 0 && nested_level == 0) {
    thread_obj->prev_cct = NULL;
  }

  TMSG(TORCH_MONITOR, "Exit forward level %u state %p", nested_level, thread_obj->thread_state);
}


static void
backward_cleanup_callback
(
 const torch_monitor_callback_data_t *callback_data,
 torch_monitor_thread_obj_t *thread_obj
)
{
  uint32_t nested_level = callback_data->data.op_data.nested_level;

  // thread_obj->prev_cct = NULL;
  // The backward compute function is invoked after the exit.
  // So don't clean up this cct

  TMSG(TORCH_MONITOR, "Exit backward level %u state %p", nested_level, thread_obj->thread_state);
}


static void
torch_monitor_callback
(
 torch_monitor_callback_site_t callback_site,
 torch_monitor_callback_data_t *callback_data
)
{
  torch_monitor_thread_obj_t *thread_obj = torch_monitor_thread_obj_get();

  if (callback_site == TORCH_MONITOR_CALLBACK_ENTER) {
    if (callback_data->domain == TORCH_MONITOR_DOMAIN_FUNCTION) {
      forward_function_callback(callback_data, thread_obj);
    } else if (callback_data->domain == TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION) {
      backward_function_callback(callback_data, thread_obj);
    }
    // TODO(Keren)
    // else if (callback_data->domain == TORCH_MONITOR_DOMAIN_MEMORY) {
    //}
  } else {
    if (callback_data->domain == TORCH_MONITOR_DOMAIN_FUNCTION) {
      forward_cleanup_callback(callback_data, thread_obj);
    } else if (callback_data->domain == TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION) {
      backward_cleanup_callback(callback_data, thread_obj);
    }
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

