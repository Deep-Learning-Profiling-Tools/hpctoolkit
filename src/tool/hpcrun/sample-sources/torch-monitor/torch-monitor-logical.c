#include "torch-monitor-logical.h"

#include <hpcrun/cct_backtrace_finalize.h>
#include <hpcrun/cct_insert_backtrace.h>
#include <hpcrun/gpu/gpu-application-thread-api.h>
#include <hpcrun/logical/common.h>
#include <hpcrun/safe-sampling.h>
#include <hpcrun/thread_data.h>
#include <torch_monitor.h>

#include "torch-monitor-op-placeholders.h"
#include "torch-monitor-thread-obj.h"

#define TORCH_MONITOR_MODULE_ID_NULL -1

#define TORCH_MONITOR_ADDITIONAL_FRAMES 2

static bool torch_monitor_native_stack_enabled = false;

static logical_metadata_store_t *torch_monitor_metadata = NULL;

static __thread int python_module_id = TORCH_MONITOR_MODULE_ID_NULL;

bool torch_monitor_native_stack_status_get() {
  return torch_monitor_native_stack_enabled;
}

int torch_monitor_python_module_id_get() {
  // Assuming python is not loaded and unloaded
  return python_module_id;
}

ip_normalized_t torch_monitor_function_ip(const char *function_name) {
  const char *metadata_path =
      hpcrun_logical_metadata_path_get(torch_monitor_metadata);
  uint32_t fid =
      hpcrun_logical_metadata_fid(torch_monitor_metadata, function_name,
                                  LOGICAL_MANGLING_NONE, metadata_path, 0);
  ip_normalized_t ip_norm =
      hpcrun_logical_metadata_ipnorm(torch_monitor_metadata, fid, 0);

  TORCH_MONITOR_MSG("function name %s metapath %s", function_name,
                    metadata_path);

  return ip_norm;
}

static cct_node_t *backtrace_phase_insert(
    torch_monitor_thread_obj_t *thread_obj, cct_node_t *cct) {
  if (thread_obj->domain == TORCH_MONITOR_DOMAIN_FUNCTION) {
    cct = torch_monitor_op_cct_insert(
        cct, torch_monitor_op_placeholder_type_forward);
  } else if (thread_obj->domain == TORCH_MONITOR_DOMAIN_BACKWARD_FUNCTION) {
    cct = torch_monitor_op_cct_insert(
        cct, torch_monitor_op_placeholder_type_backward);
  }

  return cct;
}

static cct_node_t *backtrace_function_insert(
    torch_monitor_thread_obj_t *thread_obj, cct_node_t *cct) {
  cct = hpcrun_cct_insert_ip_norm(cct, thread_obj->function_ip_norm, true);
  cct = backtrace_phase_insert(thread_obj, cct);

  return cct;
}

static bool btbuf_ensure(size_t num_frames) {
  // Allocate additional frames
  thread_data_t *td = hpcrun_get_thread_data();
  size_t original_size = td->btbuf_end - td->btbuf_beg;

  while ((td->btbuf_end - td->btbuf_beg) < num_frames) {
    td->btbuf_cur = hpcrun_expand_btbuf();
    td->btbuf_sav = td->btbuf_end;
  }

  size_t cur_size = td->btbuf_end - td->btbuf_beg;
  TORCH_MONITOR_MSG("Expand btbuf from %lu to %lu", original_size, cur_size);

  return cur_size != original_size;
}

static void python_callpath_unwind(torch_monitor_thread_obj_t *thread_obj,
                                   frame_t **btbuf_cur) {
  TORCH_MONITOR_MSG(
      "Frame start ==================================================");

  int i;
  for (i = 0; i < thread_obj->python_cur_num_states; ++i) {
    torch_monitor_python_state_t *python_state = &thread_obj->python_states[i];
    uint32_t fid = hpcrun_logical_metadata_fid(
        torch_monitor_metadata, python_state->function_name,
        LOGICAL_MANGLING_NONE, python_state->file_name,
        python_state->function_first_lineno);
    ip_normalized_t ip_norm = hpcrun_logical_metadata_ipnorm(
        torch_monitor_metadata, fid, python_state->lineno);

    TORCH_MONITOR_MSG("file name %s function name %s->ip norm %u %p",
                      python_state->file_name, python_state->function_name,
                      ip_norm.lm_id, ip_norm.lm_ip);

    (*btbuf_cur)->ip_norm = ip_norm;
    (*btbuf_cur)++;
  }

  TORCH_MONITOR_MSG(
      "Frame end ==================================================");
}

cct_node_t *torch_monitor_backtrace2cct(cct_bundle_t *cct, int metric_id,
                                        hpcrun_metricVal_t metric_incr) {
  TORCH_MONITOR_MSG("Enter torch_monitor_backtrace2cct");

  torch_monitor_thread_obj_t *thread_obj = torch_monitor_thread_obj_get();
  cct_node_t *node = NULL;

  if (thread_obj->thread_state == TORCH_MONITOR_THREAD_STATE_NONE) {
    return NULL;
  }

  if (thread_obj->prev_cct != NULL) {
    TORCH_MONITOR_MSG("Fast path backtrace2cct");

    // If this op happens between op_enter and op_exit, we use cached cct node
    node = thread_obj->prev_cct;
    metric_data_list_t *metric_set = hpcrun_reify_metric_set(node, metric_id);
    metric_upd_proc_t *upd_proc = hpcrun_get_metric_proc(metric_id);
    if (upd_proc) {
      upd_proc(metric_id, metric_set, metric_incr);
    }
  } else {
    if (thread_obj->function_cct != NULL) {
      TORCH_MONITOR_MSG("Backward fast path backtrace2cct");

      node = backtrace_phase_insert(thread_obj, thread_obj->function_cct);
    } else {
      TORCH_MONITOR_MSG("Forward slow path backtrace2cct");

      // Otherwise, we unwind python call path
      thread_data_t *td = hpcrun_get_thread_data();

      td->btbuf_cur = td->btbuf_beg;  // innermost
      td->btbuf_sav = td->btbuf_end;  // FIXME: is it needed?

      // Update python states
      torch_monitor_python_state_get(thread_obj->python_max_num_states,
                                     thread_obj->python_states,
                                     &thread_obj->python_cur_num_states);

      // Ensure btbuf is available
      btbuf_ensure(thread_obj->python_cur_num_states + 1);

      python_callpath_unwind(thread_obj, &(td->btbuf_cur));

      frame_t *bt_beg = td->btbuf_beg;      // innermost, inclusive
      frame_t *bt_end = td->btbuf_cur - 1;  // outermost, inclusive

      cct_node_t *cct_cursor = cct->tree_root;
      node = hpcrun_cct_insert_backtrace_w_metric(cct_cursor, metric_id, bt_end,
                                                  bt_beg, metric_incr, NULL);
      // Insert both funtion and phase nodes
      node = backtrace_function_insert(thread_obj, node);
    }

    // Cache prev_node in any case
    thread_obj->prev_cct = node;
  }

  TORCH_MONITOR_MSG("Exit torch_monitor_backtrace2cct");

  return node;
}

static frame_t *python_start_frame_get(backtrace_info_t *bt) {
  // bottom frames ... top frames
  // bt->begin (inclusive) ... bt->last (inclusive)
  // Assuming the call stack has python interpreters on the top
  frame_t *btbuf_cur = bt->begin;  // Inclusive
  while (btbuf_cur != bt->last) {
    uint16_t lm_id = btbuf_cur->ip_norm.lm_id;

    if (python_module_id == TORCH_MONITOR_MODULE_ID_NULL) {
      load_module_t *module = hpcrun_loadmap_findById(lm_id);
      if (module != NULL && strstr(module->name, "/bin/python") != NULL) {
        python_module_id = lm_id;
      }
    }
    if (lm_id == python_module_id) {
      break;
    }
    btbuf_cur++;
  }
  return btbuf_cur;
}

static void backtrace_finalize(backtrace_info_t *bt, int is_sync) {
  torch_monitor_thread_obj_t *thread_obj = torch_monitor_thread_obj_get();

  if (thread_obj->thread_state == TORCH_MONITOR_THREAD_STATE_NONE) {
    return;
  }

  // btbuf_cur is the start of raw python frames
  frame_t *btbuf_cur = python_start_frame_get(bt);

  if (thread_obj->prev_cct != NULL) {
    TORCH_MONITOR_MSG("Fast path backtrace_finalize");

    // Python path is cached we can just adjust frame begin and last
    // btbuf_cur is a raw python frame
    bt->last = btbuf_cur - 1;  // Inclusive
  } else {
    TORCH_MONITOR_MSG("Slow path backtrace_finalize");
    // Has python frames but python module is not found
    assert(!(python_module_id == TORCH_MONITOR_MODULE_ID_NULL &&
             thread_obj->python_cur_num_states != 0));

    if (thread_obj->function_cct == NULL) {
      assert((thread_obj->thread_state & TORCH_MONITOR_THREAD_STATE_FORWARD));

      // Update python_states
      torch_monitor_python_state_get(thread_obj->python_max_num_states,
                                     thread_obj->python_states,
                                     &thread_obj->python_cur_num_states);

      size_t raw_frames = bt->last - bt->begin + 1;
      size_t raw_python_frames =
          bt->last - btbuf_cur + 1;  // btbuf_cur is a raw python frame
      size_t processed_python_frames = thread_obj->python_cur_num_states;
      size_t processed_native_frames = btbuf_cur - bt->begin;
      size_t processed_total_frames =
          processed_native_frames + processed_python_frames;

      // Nested level = 0
      // native ...     / python ...
      // bt->begin ... btbuf_cur ... bt->last
      TORCH_MONITOR_MSG(
          "raw_frames: %lu, raw_python_frames: %lu, processed_python_frames: "
          "%lu processed_native_frames: %lu, processed_total_frames: %lu\n",
          raw_frames, raw_python_frames, processed_python_frames,
          processed_native_frames, processed_total_frames);

      if (btbuf_ensure(processed_total_frames +
                       TORCH_MONITOR_ADDITIONAL_FRAMES)) {
        // btbuf was expanded
        thread_data_t *td = hpcrun_get_thread_data();
        bt->begin = td->btbuf_beg;
        btbuf_cur = bt->begin + processed_native_frames;
      }

      // Move btbuf
      memmove(btbuf_cur + TORCH_MONITOR_ADDITIONAL_FRAMES, btbuf_cur,
              sizeof(frame_t) * raw_python_frames);

      // forward node
      btbuf_cur->ip_norm = torch_monitor_op_placeholder_ip(
          torch_monitor_op_placeholder_type_forward);
      btbuf_cur++;

      // function node
      btbuf_cur->ip_norm = thread_obj->function_ip_norm;
      btbuf_cur++;

      python_callpath_unwind(thread_obj, &btbuf_cur);

      // Move native buf to the end
      bt->last = bt->begin + processed_total_frames +
                 TORCH_MONITOR_ADDITIONAL_FRAMES - 1;

      TORCH_MONITOR_MSG("Forward update btbuf");
    } else {
      // Only unwind native frames
      bt->last = btbuf_cur - 1;

      TORCH_MONITOR_MSG("Backward update btbuf");
    }
  }
}

static cct_node_t *cct_finalize(cct_bundle_t *cct,  // original cct
                                backtrace_info_t *bt,
                                cct_node_t *cursor  // refined cct by runtime
) {
  torch_monitor_thread_obj_t *thread_obj = torch_monitor_thread_obj_get();

  if (thread_obj->thread_state == TORCH_MONITOR_THREAD_STATE_NONE) {
    return cursor;
  }

  if (thread_obj->function_cct != NULL) {
    // backward
    if (thread_obj->prev_cct == NULL) {
      TORCH_MONITOR_MSG("Backward update cached prev_cct");
      // nested level = 0, update cached CCT
      cursor = backtrace_phase_insert(thread_obj, thread_obj->function_cct);
      // We can cache backward prev_cct now
      thread_obj->prev_cct = cursor;
    } else {
      TORCH_MONITOR_MSG("Backward get cached prev_cct");
      // nested level != 0, use cached CCT
      cursor = thread_obj->prev_cct;
    }
  } else if (thread_obj->prev_cct != NULL) {
    TORCH_MONITOR_MSG("Forward get cached prev_cct");
    cursor = thread_obj->prev_cct;
  } else {
    TORCH_MONITOR_MSG("Forward no cached prev_cct");
  }
  // else forward
  // nested level = 0
  // TODO(Keren): cache forward cct

  return cursor;
}

void torch_monitor_logical_register(bool native_stack) {
  TORCH_MONITOR_MSG("Enter torch_monitor_logical_register");

  torch_monitor_native_stack_enabled = native_stack;
  hpcrun_logical_metadata_register(&torch_monitor_metadata, "torch_monitor");

  if (torch_monitor_native_stack_enabled == true) {
    static cct_backtrace_finalize_entry_t backtrace_entry = {
        backtrace_finalize};
    static bool torch_monitor_backtrace_registered = false;
    if (!torch_monitor_backtrace_registered) {
      cct_backtrace_finalize_register(&backtrace_entry);
      cct_cursor_finalize_register(cct_finalize);
      torch_monitor_backtrace_registered = true;
    }
  }

  TORCH_MONITOR_MSG("Exit torch_monitor_logical_register");
}

void torch_monitor_logical_unregister(void) {
  TORCH_MONITOR_MSG("Enter torch_monitor_logical_unregister");

  torch_monitor_native_stack_enabled = false;

  TORCH_MONITOR_MSG("Exit torch_monitor_logical_unregister");
}
