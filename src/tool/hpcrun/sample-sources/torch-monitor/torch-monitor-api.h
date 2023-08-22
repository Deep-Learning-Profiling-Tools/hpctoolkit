#ifndef TORCH_MONITOR_API_H
#define TORCH_MONITOR_API_H

#include <stdbool.h>

void torch_monitor_start(bool native_stack);

void torch_monitor_stop();

bool torch_monitor_status_get();

#endif  // TORCH_MONITOR_API_H
