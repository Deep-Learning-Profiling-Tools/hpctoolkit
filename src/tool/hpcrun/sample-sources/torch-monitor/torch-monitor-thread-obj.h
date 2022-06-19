#ifndef torch_monitor_thread_obj
#define torch_monitor_thread_obj

#include <torch_monitor.h>

typedef struct torch_monitor_thread_obj {
  size_t max_python_num_states;
  size_t cur_python_num_states;
  torch_monitor_thread_state_t thread_state;
  torch_monitor_python_state_t *python_states;
} torch_monitor_thread_obj_t;

torch_monitor_thread_obj_t *torch_monitor_thread_obj_get();

#endif  // torch_monitor_thread_obj
