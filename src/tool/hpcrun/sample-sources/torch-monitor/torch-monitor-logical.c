#include "torch-monitor-logical.h"

#include <hpcrun/logical-metadata.h>
#include <hpcrun/cct_backtrace_finalize.h>
#include <hpcrun/cct_insert_backtrace.h>
#include <hpcrun/safe-sampling.h>
#include <hpcrun/thread_data.h>
#include <hpcrun/gpu/gpu-application-thread-api.h>

#include <torch_monitor.h>

#include "torch-monitor-thread-obj.h"

static bool torch_monitor_native_stack_enabled = false;

static logical_metadata_store_t *torch_monitor_metadata = NULL;


bool torch_monitor_native_stack_status() {
  return torch_monitor_native_stack_enabled;
}


cct_node_t *
torch_monitor_backtrace2cct
(
 cct_bundle_t *cct,
 int metric_id,
 hpcrun_metricVal_t metric_incr
)
{
  hpcrun_safe_enter();

  torch_monitor_thread_obj_t *thread_obj = torch_monitor_thread_obj_get();
  cct_node_t *node = NULL;

  if (thread_obj->forward_cct != NULL) {
    TMSG(TORCH_MONITOR, "Fast path");

    // If this op happens between op_enter and op_exit, we used cached cct node
    node = thread_obj->forward_cct;
    metric_data_list_t *metric_set = hpcrun_reify_metric_set(node, metric_id);
    metric_upd_proc_t *upd_proc = hpcrun_get_metric_proc(metric_id);
    if (upd_proc) {
      upd_proc(metric_id, metric_set, metric_incr);
    }
  } else {
    TMSG(TORCH_MONITOR, "Slow path");

    // Otherwise, we unwind python call path
    cct_node_t* cct_cursor = cct->tree_root;
    torch_monitor_python_state_get(thread_obj->python_max_num_states, thread_obj->python_states,
      &thread_obj->python_cur_num_states);

    thread_data_t* td = hpcrun_get_thread_data();

    td->btbuf_cur = td->btbuf_beg;  // innermost
    td->btbuf_sav = td->btbuf_end;  // what is it? is it needed?

    TMSG(TORCH_MONITOR, "Frame start ==================================================");

    if (thread_obj->python_cur_num_states > 0) {
      int i;
      for (i = thread_obj->python_cur_num_states - 1; i >= 0; --i) {
        hpcrun_ensure_btbuf_avail();

        torch_monitor_python_state_t *python_state = &thread_obj->python_states[i];
        TMSG(TORCH_MONITOR, "\t%s %s:%u", python_state->file_name, python_state->function_name, python_state->lineno);
        uint32_t fid = hpcrun_logical_metadata_fid(torch_monitor_metadata,
          python_state->function_name, python_state->file_name, python_state->function_first_lineno);
        ip_normalized_t ip_norm = hpcrun_logical_metadata_ipnorm(torch_monitor_metadata,
          fid, python_state->lineno);

        td->btbuf_cur->ip_norm = ip_norm;
        td->btbuf_cur++;
      }
    }

    TMSG(TORCH_MONITOR, "Frame end ==================================================");

    frame_t* bt_beg = td->btbuf_beg;      // innermost, inclusive 
    frame_t* bt_end = td->btbuf_cur - 1;  // outermost, inclusive

    node = hpcrun_cct_insert_backtrace_w_metric(cct_cursor, metric_id, bt_end, bt_beg, metric_incr, NULL);
  }

  hpcrun_safe_exit();

  return node;
}


static void
backtrace_finalize
(
 backtrace_info_t* bt,
 int is_sync
)
{
  thread_data_t* td = hpcrun_get_thread_data();

  static __thread uint16_t python_id = 0;

  frame_t *bt_cur = td->btbuf_beg;
  frame_t *bt_end = td->btbuf_end;

  while (bt_cur != NULL) {
    uint16_t lm_id = bt_cur->ip_norm.lm_id;

    if (python_id == 0) {
      load_module_t *module = hpcrun_loadmap_findById(lm_id);
      if (module != NULL && strstr(module->name, "/python") != NULL) {
        python_id = lm_id;
      }
    }

    if (lm_id == python_id) {
      break;
    }
    bt_cur--;
  }

  // TODO(Keren): replace call stack above bt_cur
}


void
torch_monitor_logical_register
(
 bool native_stack
)
{
  torch_monitor_native_stack_enabled = native_stack;
  hpcrun_logical_metadata_register(&torch_monitor_metadata, "torch_monitor");

  if (torch_monitor_native_stack_enabled == true) {
    static cct_backtrace_finalize_entry_t backtrace_entry = {backtrace_finalize};
    static bool torch_monitor_backtrace_registered = false;
    if(!torch_monitor_backtrace_registered) {
      cct_backtrace_finalize_register(&backtrace_entry);
      torch_monitor_backtrace_registered = true;
    }
  }
}

void
torch_monitor_logical_unregister
(
 void
)
{
  torch_monitor_native_stack_enabled = false;
  hpcrun_logical_metadata_cleanup(torch_monitor_metadata);
}
