// -*-Mode: C++;-*- // technically C99

// * BeginRiceCopyright *****************************************************
//
// --------------------------------------------------------------------------
// Part of HPCToolkit (hpctoolkit.org)
//
// Information about sources of support for research and development of
// HPCToolkit is at 'hpctoolkit.org' and in 'README.Acknowledgments'.
// --------------------------------------------------------------------------
//
// Copyright ((c)) 2002-2020, Rice University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright
//   notice, this list of conditions and the following disclaimer.
//
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of Rice University (RICE) nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
//
// This software is provided by RICE and contributors "as is" and any
// express or implied warranties, including, but not limited to, the
// implied warranties of merchantability and fitness for a particular
// purpose are disclaimed. In no event shall RICE or contributors be
// liable for any direct, indirect, incidental, special, exemplary, or
// consequential damages (including, but not limited to, procurement of
// substitute goods or services; loss of use, data, or profits; or
// business interruption) however caused and on any theory of liability,
// whether in contract, strict liability, or tort (including negligence
// or otherwise) arising in any way out of the use of this software, even
// if advised of the possibility of such damage.
//
// ******************************************************* EndRiceCopyright *


#include "torch-monitor-forward-cct-map.h"

#include <lib/prof-lean/splay-macros.h>
#include <lib/prof-lean/spinlock.h>
#include <hpcrun/memory/hpcrun-malloc.h>

//******************************************************************************
// type declarations
//******************************************************************************

typedef struct splay_forward_node_t {
  struct splay_forward_node_t *left;
  struct splay_forward_node_t *right;
  forward_key_t key;
} splay_forward_node_t;


typedef enum splay_order_t {
  splay_inorder = 1,
  splay_allorder = 2
} splay_order_t;


typedef enum splay_visit_t {
  splay_preorder_visit = 1,
  splay_inorder_visit = 2,
  splay_postorder_visit = 3
} splay_visit_t;


typedef void (*splay_fn_t)
(
 splay_forward_node_t *node,
 splay_visit_t order,
 void *arg
);


bool forward_cmp_lt(forward_key_t left, forward_key_t right);
bool forward_cmp_gt(forward_key_t left, forward_key_t right);

#define forward_builtin_lt(a, b) (forward_cmp_lt(a, b))
#define forward_builtin_gt(a, b) (forward_cmp_gt(a, b))

#define FORWARD_SPLAY_TREE(type, root, key, value, left, right)	\
  GENERAL_SPLAY_TREE(type, root, key, value, value, left, right, forward_builtin_lt, forward_builtin_gt)

//*****************************************************************************
// macros 
//*****************************************************************************

#define splay_macro_body_ignore(x) ;
#define splay_macro_body_show(x) x

// declare typed interface functions 
#define typed_splay_declare(type)		\
  typed_splay(type, splay_macro_body_ignore)

// implementation of typed interface functions 
#define typed_splay_impl(type)			\
  typed_splay(type, splay_macro_body_show)

// routine name for a splay operation
#define splay_op(op) \
  splay_forward ## _ ## op

#define typed_splay_node(type) \
  type ## _ ## splay_forward_node_t

// routine name for a typed splay operation
#define typed_splay_op(type, op) \
  type ## _  ## op

// insert routine name for a typed splay 
#define typed_splay_splay(type) \
  typed_splay_op(type, splay)

// insert routine name for a typed splay 
#define typed_splay_insert(type) \
  typed_splay_op(type, insert)

// lookup routine name for a typed splay 
#define typed_splay_lookup(type) \
  typed_splay_op(type, lookup)

// delete routine name for a typed splay 
#define typed_splay_delete(type) \
  typed_splay_op(type, delete)

// forall routine name for a typed splay 
#define typed_splay_forall(type) \
  typed_splay_op(type, forall)

// count routine name for a typed splay 
#define typed_splay_count(type) \
  typed_splay_op(type, count)

// define typed wrappers for a splay type 
#define typed_splay(type, macro) \
  static bool \
  typed_splay_insert(type) \
  (typed_splay_node(type) **root, typed_splay_node(type) *node)	\
  macro({ \
    return splay_op(insert)((splay_forward_node_t **) root, (splay_forward_node_t *) node); \
  }) \
\
  static typed_splay_node(type) * \
  typed_splay_lookup(type) \
  (typed_splay_node(type) **root, forward_key_t key) \
  macro({ \
    typed_splay_node(type) *result = (typed_splay_node(type) *) \
      splay_op(lookup)((splay_forward_node_t **) root, key); \
    return result; \
  })

#define typed_splay_alloc(free_list, splay_node_type)		\
  (splay_node_type *) splay_forward_alloc_helper		\
  ((splay_forward_node_t **) free_list, sizeof(splay_node_type))	

#define typed_splay_free(free_list, node)			\
  splay_forward_free_helper					\
  ((splay_forward_node_t **) free_list,				\
   (splay_forward_node_t *) node)

//******************************************************************************
// interface operations
//******************************************************************************

bool
forward_cmp_gt
(
 forward_key_t left,
 forward_key_t right
)
{
  if (left.forward_thread_id > right.forward_thread_id) {
    return true;
  } else if (left.forward_thread_id == right.forward_thread_id) {
    return left.sequence_number > right.sequence_number;
  }
  return false;
}


bool
forward_cmp_lt
(
 forward_key_t left,
 forward_key_t right
)
{
  if (left.forward_thread_id < right.forward_thread_id) {
    return true;
  } else if (left.forward_thread_id == right.forward_thread_id) {
    return left.sequence_number < right.sequence_number;
  }
  return false;
}


bool
forward_cmp_eq
(
 forward_key_t left,
 forward_key_t right
)
{
  return left.forward_thread_id == right.forward_thread_id && left.sequence_number == right.sequence_number;
}


splay_forward_node_t *
splay_splay
(
 splay_forward_node_t *root,
 forward_key_t splay_key
)
{
  FORWARD_SPLAY_TREE(splay_forward_node_t, root, splay_key, key, left, right);

  return root;
}


bool
splay_forward_insert
(
 splay_forward_node_t **root,
 splay_forward_node_t *node
)
{
  node->left = node->right = NULL;

  if (*root != NULL) {
    *root = splay_splay(*root, node->key);
    
    if (forward_cmp_lt(node->key, (*root)->key)) {
      node->left = (*root)->left;
      node->right = *root;
      (*root)->left = NULL;
    } else if (forward_cmp_gt(node->key, (*root)->key)) {
      node->left = *root;
      node->right = (*root)->right;
      (*root)->right = NULL;
    } else {
      // key already present in the tree
      return true; // insert failed 
    }
  } 

  *root = node;

  return true; // insert succeeded 
}


splay_forward_node_t *
splay_forward_lookup
(
 splay_forward_node_t **root,
 forward_key_t key
)
{
  splay_forward_node_t *result = 0;

  *root = splay_splay(*root, key);

  if (*root && forward_cmp_eq((*root)->key, key)) {
    result = *root;
  }

  return result;
}


splay_forward_node_t *
splay_forward_alloc_helper
(
 splay_forward_node_t **free_list, 
 size_t size
)
{
  splay_forward_node_t *first = *free_list; 

  if (first) { 
    *free_list = first->left;
  } else {
    first = (splay_forward_node_t *) hpcrun_malloc_safe(size);
  }

  memset(first, 0, size); 

  return first;
}


void
splay_forward_free_helper
(
 splay_forward_node_t **free_list, 
 splay_forward_node_t *e 
)
{
  e->left = *free_list;
  *free_list = e;
}

//******************************************************************************
// private operations
//******************************************************************************

#define st_insert				\
  typed_splay_insert(forward)

#define st_lookup				\
  typed_splay_lookup(forward)

#define st_delete				\
  typed_splay_delete(forward)

#define st_forall				\
  typed_splay_forall(forward)

#define st_count				\
  typed_splay_count(forward)

#define st_alloc(free_list)			\
  typed_splay_alloc(free_list, typed_splay_node(forward))

#define st_free(free_list, node)		\
  typed_splay_free(free_list, node)

#undef typed_splay_node
#define typed_splay_node(forward) torch_monitor_forward_cct_map_entry_t

struct torch_monitor_forward_cct_map_entry_s {
  struct torch_monitor_forward_cct_map_entry_s *left;
  struct torch_monitor_forward_cct_map_entry_s *right;
  forward_key_t key;
  cct_node_t *cct;
};

// TODO(Keren): pass forward records to the backward phase
static spinlock_t map_lock = SPINLOCK_UNLOCKED;
static torch_monitor_forward_cct_map_entry_t *map_root = NULL;
static torch_monitor_forward_cct_map_entry_t *free_list = NULL;

typed_splay_impl(forward)

static torch_monitor_forward_cct_map_entry_t *
torch_monitor_forward_cct_map_new
(
 forward_key_t key,
 cct_node_t *cct
)
{
  torch_monitor_forward_cct_map_entry_t *entry = st_alloc(&free_list);
  
  entry->key = key;
  entry->cct = cct;

  return entry;
}


void
torch_monitor_forward_cct_map_insert
(
 forward_key_t key,
 cct_node_t *cct
)
{
  spinlock_lock(&map_lock);

  torch_monitor_forward_cct_map_entry_t *entry = st_lookup(&map_root, key);

  if (entry == NULL) {
    entry = torch_monitor_forward_cct_map_new(key, cct);
    st_insert(&map_root, entry);
  }

  spinlock_unlock(&map_lock);
}


torch_monitor_forward_cct_map_entry_t *
torch_monitor_forward_cct_map_lookup
(
 forward_key_t key
)
{
  torch_monitor_forward_cct_map_entry_t *entry = NULL;

  spinlock_lock(&map_lock);

  entry = st_lookup(&map_root, key);

  spinlock_unlock(&map_lock);

  return entry;
}


void
torch_monitor_forward_cct_map_entry_cct_update
(
 torch_monitor_forward_cct_map_entry_t *entry,
 cct_node_t *cct
)
{
  entry->cct = cct;
}


cct_node_t *
torch_monitor_forward_cct_map_entry_cct_get
(
 torch_monitor_forward_cct_map_entry_t *entry
)
{
  return entry->cct;
}
