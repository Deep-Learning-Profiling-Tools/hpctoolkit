#include "torch-monitor-thread-obj.h"

#include <stdint.h>

#include <hpcrun/memory/hpcrun-malloc.h>

const static size_t MAX_NUM_STATES = 100;

static __thread torch_monitor_thread_obj_t *thread_obj;


static void torch_monitor_thread_obj_create(torch_monitor_thread_obj_t **obj) {
  *obj = hpcrun_malloc_safe(sizeof(torch_monitor_thread_obj_t));
  (*obj)->thread_state = 0;
  (*obj)->python_states_cached = false;
  (*obj)->python_max_num_states = MAX_NUM_STATES;
  (*obj)->python_cur_num_states = 0;
  (*obj)->python_states = hpcrun_malloc_safe(sizeof(torch_monitor_python_state_t) * MAX_NUM_STATES);
}


torch_monitor_thread_obj_t *torch_monitor_thread_obj_get() {
  if (thread_obj == NULL) {
    torch_monitor_thread_obj_create(&thread_obj);
  }
  return thread_obj;
}
