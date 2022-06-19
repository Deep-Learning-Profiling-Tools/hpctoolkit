#ifndef logical_metadata_h
#define logical_metadata_h

#include <stdint.h>

#include <hpcrun/cct/cct.h>

#include <lib/prof-lean/spinlock.h>
#include <lib/prof-lean/stdatomic.h>

typedef struct logical_metadata_store_hashentry logical_metadata_store_hashentry_t;
typedef struct logical_metadata_store logical_metadata_store_t;

// Register a logical metadata store with the given identifier
void hpcrun_logical_metadata_register(logical_metadata_store_t **store, const char* generator);

// Generate a new load module id for a metadata store. Expensive
// Requires that the lock is held, in practice just use the wrapper below
void hpcrun_logical_metadata_generate_lmid(logical_metadata_store_t* store);

// Get the load module id for a metadata store. Caches properly
uint16_t hpcrun_logical_metadata_lmid(logical_metadata_store_t* store);

// Get a unique identifier for:
//  - (NULL, NULL, <ignored>): Unknown logical region (always 0)
//  - (NULL, "file", <ignored>): A source file "file" with no known function information
//  - ("func", NULL, <ignored>): A function "func" with no known source file
//  - ("func", "file", lineno): A function "func" defined in "file" at line `lineno`
// This is nearly useless alone, pass to hpcrun_logical_metadata_ipnorm
uint32_t hpcrun_logical_metadata_fid(logical_metadata_store_t*,
  const char* func, const char* file, uint32_t lineno);

// Compose a full normalized ip for a particular line number within a logical function/file.
ip_normalized_t hpcrun_logical_metadata_ipnorm(
  logical_metadata_store_t* store, uint32_t fid, uint32_t lineno);

// Write metadata infor to disk
void hpcrun_logical_metadata_cleanup(logical_metadata_store_t* store);

#endif  // logical_metadata_h

