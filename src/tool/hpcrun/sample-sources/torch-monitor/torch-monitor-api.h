#ifndef torch_monitor_api_h
#define torch_monitor_api_h

#include <stdbool.h>

void torch_monitor_start(bool native_stack);

void torch_monitor_stop();

bool torch_monitor_status();

#endif  // torch_monitor_api_h
