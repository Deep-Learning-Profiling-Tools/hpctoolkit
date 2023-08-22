#ifndef HPCRUN_CONTROL_KNOB_H
#define HPCRUN_CONTROL_KNOB_H

typedef enum{
  ck_int = 0,
  ck_float = 1,
  ck_string = 2
} control_knob_type;

// Add to register from Command line
void control_knob_init();

// Add to register from sample source
void control_knob_register(char *name, char *value, control_knob_type type);

int control_knob_value_get_int(char *in, int *value);
int control_knob_value_get_float(char *in, float *value);
int control_knob_value_get_string(char *in, char **value);

#endif
