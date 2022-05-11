#include "torch-monitor-logical.h"

#include <hpcrun/logical-metadata.h>
#include <hpcrun/cct_backtrace_finalize.h>
#include <hpcrun/cct_insert_backtrace.h>
#include <hpcrun/safe-sampling.h>
#include <hpcrun/thread_data.h>
#include <hpcrun/gpu/gpu-application-thread-api.h>

#include <torch_monitor.h>

#include "torch-monitor-thread-obj.h"

#define TORCH_MONITOR_MODULE_ID_NULL -1

static bool torch_monitor_native_stack_enabled = false;

static logical_metadata_store_t *torch_monitor_metadata = NULL;

bool torch_monitor_native_stack_status() {
  return torch_monitor_native_stack_enabled;
}


cct_node_t *
torch_monitor_backtrace_function_insert
(
 cct_node_t *cct,
 const char *function_name
)
{

  const char *metadata_path = hpcrun_logical_metadata_path_get(torch_monitor_metadata);
  uint32_t fid = hpcrun_logical_metadata_fid(torch_monitor_metadata, function_name, metadata_path, 0);
  ip_normalized_t ip_norm = hpcrun_logical_metadata_ipnorm(torch_monitor_metadata, fid, 0);
  return hpcrun_cct_insert_ip_norm(cct, ip_norm, true);
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

  if (thread_obj->prev_cct != NULL) {
    TMSG(TORCH_MONITOR, "Fast path backtrace2cct");

    // If this op happens between op_enter and op_exit, we used cached cct node
    node = thread_obj->prev_cct;
    metric_data_list_t *metric_set = hpcrun_reify_metric_set(node, metric_id);
    metric_upd_proc_t *upd_proc = hpcrun_get_metric_proc(metric_id);
    if (upd_proc) {
      upd_proc(metric_id, metric_set, metric_incr);
    }
  } else {
    TMSG(TORCH_MONITOR, "Slow path backtrace2cct");

    // Otherwise, we unwind python call path
    cct_node_t* cct_cursor = cct->tree_root;
    torch_monitor_python_state_get(thread_obj->python_max_num_states, thread_obj->python_states,
      &thread_obj->python_cur_num_states);

    thread_data_t* td = hpcrun_get_thread_data();

    td->btbuf_cur = td->btbuf_beg;  // innermost
    td->btbuf_sav = td->btbuf_end;  // what is it? is it needed?

    TMSG(TORCH_MONITOR, "Frame start ==================================================");

    int i;
    for (i = 0; i < thread_obj->python_cur_num_states; ++i) {
      hpcrun_ensure_btbuf_avail();

      torch_monitor_python_state_t *python_state = &thread_obj->python_states[i];
      uint32_t fid = hpcrun_logical_metadata_fid(torch_monitor_metadata,
        python_state->function_name, python_state->file_name, python_state->function_first_lineno);
      ip_normalized_t ip_norm = hpcrun_logical_metadata_ipnorm(torch_monitor_metadata,
        fid, python_state->lineno);

      td->btbuf_cur->ip_norm = ip_norm;
      td->btbuf_cur++;
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
  static __thread int python_id = TORCH_MONITOR_MODULE_ID_NULL;

  // bottom frames ... top frames
  // bt->begin (inclusive) ... bt->last (inclusive)
  frame_t *bt_cur = bt->begin;  // Inclusive

  // Assuming the call stack has python interpreters on the top
  while (bt_cur != bt->last) {
    uint16_t lm_id = bt_cur->ip_norm.lm_id;

    if (python_id == TORCH_MONITOR_MODULE_ID_NULL) {
      load_module_t *module = hpcrun_loadmap_findById(lm_id);
      if (module != NULL && strstr(module->name, "/bin/python") != NULL) {
        python_id = lm_id;
      }
    }

    if (lm_id == python_id) {
      break;
    }
    bt_cur++;
  }

  // Always slow path
  // TODO(Keren): replace call stack above bt_cur
  // TODO(Keren): cache python call path
  torch_monitor_thread_obj_t *thread_obj = torch_monitor_thread_obj_get();
  torch_monitor_python_state_get(thread_obj->python_max_num_states, thread_obj->python_states,
    &thread_obj->python_cur_num_states);
  // Has python frames but python module is not found
  assert(!(python_id == TORCH_MONITOR_MODULE_ID_NULL && thread_obj->python_cur_num_states != 0));

  size_t raw_frames = bt->last - bt->begin + 1;
  size_t raw_python_frames = bt->last - bt_cur;  // bt_cur is a native frame
  size_t processed_python_frames = thread_obj->python_cur_num_states;
  size_t processed_native_frames = bt_cur - bt->begin + 1;
  size_t processed_total_frames = processed_native_frames + processed_python_frames;

  TMSG(TORCH_MONITOR, "raw_frames: %lu, raw_python_frames: %lu, processed_python_frames: %lu processed_native_frames: %lu, processed_total_frames: %lu\n", raw_frames, raw_python_frames, processed_python_frames, processed_native_frames, processed_total_frames);

  thread_data_t* td = hpcrun_get_thread_data();
#if 0
  td->btbuf_cur = td->btbuf_beg;  // innermost

  int i;
  for (i = 0; i < total_frames; ++i) {
    td->btbuf_cur++;
    hpcrun_ensure_btbuf_avail();
  }

  td->btbuf_cur = td->btbuf_beg;  // innermost
  td->btbuf_sav = td->btbuf_end;  // what is it? is it needed?

  TMSG(TORCH_MONITOR, "Frame start ==================================================");

  // move native buf to the end
  memmove(td->btbuf_beg + python_frames, bt_cur, sizeof(frame_t) * native_frames);
  for (i = 0; i < thread_obj->python_cur_num_states; ++i) {
    torch_monitor_python_state_t *python_state = &thread_obj->python_states[i];
    uint32_t fid = hpcrun_logical_metadata_fid(torch_monitor_metadata,
      python_state->function_name, python_state->file_name, python_state->function_first_lineno);
    TMSG(TORCH_MONITOR, "\t%s %s:%u fid: %u", python_state->file_name, python_state->function_name, python_state->lineno, fid);
    ip_normalized_t ip_norm = hpcrun_logical_metadata_ipnorm(torch_monitor_metadata,
      fid, python_state->lineno);

    td->btbuf_cur->ip_norm = ip_norm;
    td->btbuf_cur++;
  }
#endif
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
