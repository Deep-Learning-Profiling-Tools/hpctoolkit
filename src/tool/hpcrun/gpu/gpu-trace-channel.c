//******************************************************************************
// global includes
//******************************************************************************

#include <string.h>
#include <pthread.h>



//******************************************************************************
// local includes
//******************************************************************************

#include <lib/prof-lean/bichannel.h>

#include <hpcrun/memory/hpcrun-malloc.h>

#include "gpu-trace.h"
#include "gpu-trace-channel.h"
#include "gpu-trace-item.h"



//******************************************************************************
// macros
//******************************************************************************

#define CHANNEL_FILL_COUNT 100


#undef typed_bichannel
#undef typed_stack_elem

#define typed_bichannel(x) gpu_trace_channel_t
#define typed_stack_elem(x) gpu_trace_item_t

// define macros that simplify use of trace channel API 
#define channel_init  \
  typed_bichannel_init(gpu_trace_item_t)

#define channel_pop   \
  typed_bichannel_pop(gpu_trace_item_t)

#define channel_push  \
  typed_bichannel_push(gpu_trace_item_t)

#define channel_reverse \
  typed_bichannel_reverse(gpu_trace_item_t)

#define channel_steal \
  typed_bichannel_steal(gpu_trace_item_t)



//******************************************************************************
// type declarations
//******************************************************************************

typedef struct gpu_trace_channel_t {
  bistack_t bistacks[2];
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  uint64_t count;
} gpu_trace_channel_t;



//******************************************************************************
// private functions
//******************************************************************************

// implement bidirectional channels for traces
typed_bichannel_impl(gpu_trace_item_t)

static void
gpu_trace_channel_signal_consumer_when_full
(
 gpu_trace_channel_t *trace_channel
)
{
  if (trace_channel->count++ > CHANNEL_FILL_COUNT) {
    trace_channel->count = 0;
    gpu_trace_channel_signal_consumer(trace_channel);
  }
}



//******************************************************************************
// interface functions
//******************************************************************************

gpu_trace_channel_t *
gpu_trace_channel_alloc
(
 void
)
{
  gpu_trace_channel_t *channel = 
    hpcrun_malloc_safe(sizeof(gpu_trace_channel_t));

  memset(channel, 0, sizeof(gpu_trace_channel_t));

  channel_init(channel);

  pthread_mutex_init(&channel->mutex, NULL);
  pthread_cond_init(&channel->cond, NULL);

  return channel;
}


void
gpu_trace_channel_produce
(
 gpu_trace_channel_t *channel,
 uint64_t start,
 uint64_t end,
 cct_node_t *call_path_leaf
)
{
  gpu_trace_item_t *ti = gpu_trace_item_alloc(channel);

  gpu_trace_item_produce(ti, start, end, call_path_leaf);

  channel_push(channel, bichannel_direction_forward, ti);
  
  gpu_trace_channel_signal_consumer_when_full(channel);
}


void
gpu_trace_channel_consume
(
 gpu_trace_channel_t *channel,
 thread_data_t *td, 
 gpu_trace_item_consume_fn_t trace_item_consume
)
{
  // steal elements previously pushed by the producer
  channel_steal(channel, bichannel_direction_forward);

  // reverse them so that they are in FIFO order
  channel_reverse(channel, bichannel_direction_forward);

  // consume all elements enqueued before this function was called
  for (;;) {
    gpu_trace_item_t *ti = channel_pop(channel, bichannel_direction_forward);
    if (!ti) break;
    gpu_trace_item_consume(trace_item_consume, td, ti);
    gpu_trace_item_free(channel, ti);
  }
}


void
gpu_trace_channel_await
(
 gpu_trace_channel_t *channel
)
{
  pthread_cond_wait(&channel->cond, &channel->mutex);
}


void
gpu_trace_channel_signal_consumer
(
 gpu_trace_channel_t *trace_channel
)
{
  pthread_cond_signal(&trace_channel->cond);
}

