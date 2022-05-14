#include "logical-metadata.h"

#include <hpcrun/files.h>
#include <hpcrun/memory/hpcrun-malloc.h>
#include <hpcrun/messages/messages.h>

#include <sys/stat.h>
#include <stdint.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>

// -------------------------------------
// Forward declaration
// -------------------------------------

static size_t string_hash(const char* data);

static size_t int_hash(uint32_t x);

static logical_metadata_store_hashentry_t *hashtable_probe(
  logical_metadata_store_t* store, const logical_metadata_store_hashentry_t* needle);

static void hashtable_grow(logical_metadata_store_t* store);

// --------------------------------------
// Logical load module management
// --------------------------------------

typedef struct logical_metadata_store_hashentry {
  char* funcname;  // funcname used for this entry
  char* filename;  // filename used for this entry
  uint32_t lineno; // lineno used for this entry
  size_t hash;  // Full hash value for this entry
  uint32_t id;  // value: Identifier for this key
} logical_metadata_store_hashentry_t;

// There will generally be one of these per logical context generator
// The actual initialization is handled in hpcrun_logical_metadata_register
typedef struct logical_metadata_store {
  // Lock used to protect everything in here
  spinlock_t lock;

  // Next identifier to be allocated somewhere
  uint32_t nextid;
  // Hash table for function/file name identifiers
  struct logical_metadata_store_hashentry* idtable;
  // Current size of the hash table, must always be a power of 2
  size_t tablesize;

  // Generator identifier for this store
  const char* generator;
  // Load module identifier used to refer to this particular metadata store
  // Use hpcrun_logical_metadata_lmid to get the correct value for this
  _Atomic(uint16_t) lm_id;
  // Path to the metadata storage file
  char* path;

  // Pointer to the next metadata storage (for cleanup by fini())
  struct logical_metadata_store* next;
} logical_metadata_store_t;

// Pointer to the head of metadata storage
static logical_metadata_store_t *metadata;

static void store_path_init(logical_metadata_store_t* store) {
  // Storage for the path we will be generating
  // <output dir> + /logical/ + <generator> + . + <8 random hex digits> + \0
  store->path = hpcrun_malloc_safe(strlen(hpcrun_files_output_directory())
                              + 9 + strlen(store->generator) + 1 + 8 + 1);
  if(store->path == NULL)
    hpcrun_abort("hpcrun: error allocating space for logical metadata path");

  // First make sure the directory is created
  char* next = store->path+sprintf(store->path, "%s/logical", hpcrun_files_output_directory());
  int ret = mkdir(store->path, 0755);
  if(ret != 0 && errno != EEXIST) {
    hpcrun_abort("hpcrun: error creating logical metadata output directory `%s`: %s",
                 store->path, strerror(errno));
  }

  next += sprintf(next, "/%s.", store->generator);
  int fd = -1;
  do {
    // Create the file where all the bits will be dumped
    // The last part is just randomly generated to not conflict
    sprintf(next, "%08lx", random());
    fd = open(store->path, O_WRONLY | O_EXCL | O_CREAT, 0644);
    if(fd == -1) {
      if(errno == EEXIST) continue;  // Try again
      hpcrun_abort("hpcrun: error creating logical metadata output `%s`: %s",
                   store->path, strerror(errno));
    }
    break;
  } while(1);
  close(fd);

}


void hpcrun_logical_metadata_register(logical_metadata_store_t **store, const char* generator) {
  if (*store == NULL) {
    *store = hpcrun_malloc_safe(sizeof(logical_metadata_store_t));
  }

  spinlock_init(&(*store)->lock);
  (*store)->nextid = 1;  // 0 is reserved for the logical unknown
  (*store)->idtable = NULL;
  (*store)->tablesize = 0;
  (*store)->generator = strdup(generator);
  atomic_init(&(*store)->lm_id, 0);
  store_path_init(*store);
  (*store)->next = metadata;
  metadata = (*store);
}

ip_normalized_t hpcrun_logical_metadata_ipnorm(
    logical_metadata_store_t* store, uint32_t fid, uint32_t lineno) {
  ip_normalized_t ip = {
    .lm_id = hpcrun_logical_metadata_lmid(store), .lm_ip = fid,
  };
  ip.lm_ip = (ip.lm_ip << 32) + lineno;
  return ip;
}

uint16_t hpcrun_logical_metadata_lmid(logical_metadata_store_t* store) {
  uint16_t ret = atomic_load_explicit(&store->lm_id, memory_order_relaxed);
  if(ret == 0) {
    spinlock_lock(&store->lock);
    ret = atomic_load_explicit(&store->lm_id, memory_order_relaxed);
    if(ret == 0) {
      hpcrun_logical_metadata_generate_lmid(store);
      ret = atomic_load_explicit(&store->lm_id, memory_order_relaxed);
    }
    spinlock_unlock(&store->lock);
  }
  return ret;
}

void hpcrun_logical_metadata_generate_lmid(logical_metadata_store_t* store) {
  if (store->path == NULL) {
    store_path_init(store);
  }
  // Register the path with the loadmap
  atomic_store_explicit(&store->lm_id, hpcrun_loadModule_add(store->path), memory_order_release);
}

// Roughly the FNV-1a hashing algorithm, simplified slightly for ease of use
static size_t string_hash(const char* data) {
  size_t sponge = (size_t)0x2002100120021001ULL;
  if(data == NULL) return sponge;
  const size_t sz = strlen(data);
  const size_t prime = _Generic((sponge),
      uint32_t: UINT32_C(0x01000193),
      uint64_t: UINT64_C(0x00000100000001b3));
  size_t i;
  for(i = 0; i < sz; i += sizeof sponge) {
    sponge ^= (size_t)(data[i]);
    sponge *= prime;
  }
  i -= sizeof sponge;
  size_t last = (size_t)0x1000000110000001ULL;
  memcpy(&last, data+i, sz-i);
  return (sponge ^ last) * prime;
}

// Integer mixer from https://stackoverflow.com/a/12996028
static size_t int_hash(uint32_t x) {
  x = ((x >> 16) ^ x) * UINT32_C(0x45d9f3b);
  x = ((x >> 16) ^ x) * UINT32_C(0x45d9f3b);
  x = (x >> 16) ^ x;
  return _Generic((size_t)0,
    uint32_t: x,
    uint64_t: ((uint64_t)x << 32) | x);
}

static logical_metadata_store_hashentry_t* hashtable_probe(
    logical_metadata_store_t* store, const logical_metadata_store_hashentry_t* needle) {
  for(size_t i = 0; i < store->tablesize/2; i++) {
    // Quadratic probe
    logical_metadata_store_hashentry_t* entry =
        &store->idtable[(needle->hash+i*i) & (store->tablesize-1)];
    if(entry->id == 0) return entry;  // Empty entry
    if(needle->hash != entry->hash) continue;
    if((needle->funcname == NULL) != (entry->funcname == NULL)) continue;
    if((needle->filename == NULL) != (entry->filename == NULL)) continue;
    if(needle->lineno != entry->lineno) continue;
    if(needle->funcname != NULL && strcmp(needle->funcname, entry->funcname) != 0)
      continue;
    if(needle->filename != NULL && strcmp(needle->filename, entry->filename) != 0)
      continue;
    return entry;  // Found it!
  }
  return NULL;  // Ran out of probes
}

static void hashtable_grow(logical_metadata_store_t* store) {
  if(store->idtable == NULL) { // First one's easy
    store->tablesize = 1<<8;  // Start off with 256
    store->idtable = hpcrun_malloc_safe(store->tablesize * sizeof store->idtable[0]);
    memset(store->idtable, 0, store->tablesize * sizeof store->idtable[0]);
    return;
  }

  size_t oldsize = store->tablesize;
  logical_metadata_store_hashentry_t* oldtable = store->idtable;
  store->tablesize *= 4;  // We want to reduce grows, to reduce our leak
  store->idtable = hpcrun_malloc_safe(store->tablesize * sizeof store->idtable[0]);
  memset(store->idtable, 0, store->tablesize * sizeof store->idtable[0]);
  for(size_t i = 0; i < oldsize; i++) {
    if(oldtable[i].id != 0) {
      logical_metadata_store_hashentry_t* e = hashtable_probe(store, &oldtable[i]);
      assert(e != NULL && "Failure while repopulating hash table!");
      *e = oldtable[i];
    }
  }
  // free(oldtable);  // XXX(Keren): hpcrun_malloc memory isn't freeable
}

uint32_t hpcrun_logical_metadata_fid(logical_metadata_store_t* store,
    const char* funcname, const char* filename, uint32_t lineno) {
  if(funcname == NULL && filename == NULL)
    return 0; // Specially reserved for this case
  if(funcname == NULL) lineno = 0;  // It should be ignored

  spinlock_lock(&store->lock);

  // We're looking for an entry that looks roughly like this
  logical_metadata_store_hashentry_t pattern = {
    .funcname = (char*)funcname, .filename = (char*)filename, .lineno = lineno,
    .hash = string_hash(funcname) ^ string_hash(filename) ^ int_hash(lineno),
  };

  // Probe for the entry, or where it should be.
  logical_metadata_store_hashentry_t* entry = hashtable_probe(store, &pattern);
  if(entry != NULL && entry->id != 0) {  // We have it!
    spinlock_unlock(&store->lock);
    return entry->id;
  }
  if(entry == NULL) {
    hashtable_grow(store);
    entry = hashtable_probe(store, &pattern);
    assert(entry != NULL && "Entry still not found after growth!");
  }
  *entry = pattern;

  // Make sure the entry has copies of the strings for later, and assign an id
  if(entry->funcname != NULL) {
    const char* base = entry->funcname;
    entry->funcname = hpcrun_malloc_safe(strlen(base)+1);
    strcpy(entry->funcname, base);
  }
  if(entry->filename != NULL) {
    const char* base = entry->filename;
    entry->filename = hpcrun_malloc_safe(strlen(base)+1);
    strcpy(entry->filename, base);
  }
  entry->id = store->nextid++;

  spinlock_unlock(&store->lock);
  return entry->id;
}

void hpcrun_logical_metadata_cleanup(logical_metadata_store_t* store) {
  if (store == NULL) {
    return;
  }

  FILE* f = fopen(store->path, "wb");
  if(f == NULL) return;
  fprintf(f, "HPCLOGICAL");
  for(size_t idx = 0; idx < store->tablesize; idx++) {
    logical_metadata_store_hashentry_t* entry = &store->idtable[idx];
    if(entry->id == 0) continue;
    hpcfmt_int4_fwrite(entry->id, f);
    hpcfmt_str_fwrite(entry->funcname, f);
    hpcfmt_str_fwrite(entry->filename, f);
    hpcfmt_int4_fwrite(entry->lineno, f);
    TMSG(TORCH_MONITOR, "Write id: %u funcname: %s filename: %s lineno: %u", entry->id, entry->funcname, entry->filename, entry->lineno);
  }
  fclose(f);
}

const char *hpcrun_logical_metadata_path_get(logical_metadata_store_t *store) {
  if (store == NULL) {
    return NULL;
  }
  return store->path;
}
