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
// Copyright ((c)) 2002-2022, Rice University
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

//******************************************************************************
// system includes
//******************************************************************************

#include <stdio.h>
#include <elf.h>
#include <errno.h>     // errno
#include <fcntl.h>     // open
#include <sys/stat.h>  // mkdir
#include <sys/types.h>
#include <unistd.h>
#include <linux/limits.h>  // PATH_MAX



//******************************************************************************
// local includes
//******************************************************************************

#include <include/gpu-binary.h>
#include <hpcrun/files.h>
#include <hpcrun/messages/messages.h>
#include <hpcrun/loadmap.h>
#include <lib/prof-lean/crypto-hash.h>
#include <lib/prof-lean/spinlock.h>

#ifdef ENABLE_IGC
#include <igc/ocl_igc_shared/executable_format/patch_list.h>
#endif



//******************************************************************************
// static data
//******************************************************************************

static spinlock_t binary_store_lock = SPINLOCK_UNLOCKED;
static const char elf_magic_string[] = ELFMAG;
static const uint32_t *elf_magic = (uint32_t *) elf_magic_string;

//******************************************************************************
// private operations
//******************************************************************************

bool
gpu_binary_validate_magic
(
 const char *mem_ptr,
 size_t mem_size
)
{
  if (mem_size < sizeof(uint32_t)) return false;

  uint32_t *magic = (uint32_t *) mem_ptr;

#ifdef ENABLE_IGC
  // Is this an Intel 'Patch Token' binary?
  if (*magic == MAGIC_CL) return true;
#endif

  // Is this an ELF binary?
  if (*magic == *elf_magic) return true;

  return false;
}



//******************************************************************************
// interface operations
//******************************************************************************

bool
gpu_binary_store
(
  const char *file_name,
  const void *binary,
  size_t binary_size
)
{
  // Write a file if does not exist
  bool result;
  int fd;
  errno = 0;

  spinlock_lock(&binary_store_lock);

  fd = open(file_name, O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (errno == EEXIST) {
    close(fd);
    result = true;
  } else if (fd >= 0) {
    // Success
    if (write(fd, binary, binary_size) != binary_size) {
      close(fd);
      result = false;
    } else {
      close(fd);
      result = true;
    }
  } else {
    // Failure to open is a fatal error.
    hpcrun_abort("hpctoolkit: unable to open file: '%s'", file_name);
    result = false;
  }

  spinlock_unlock(&binary_store_lock);

  return result;
}

void
gpu_binary_path_generate
(
  const char *file_name,
  char *path
)
{
  size_t used = 0;
  used += sprintf(&path[used], "%s", hpcrun_files_output_directory());
  used += sprintf(&path[used], "%s", "/" GPU_BINARY_DIRECTORY "/");
  mkdir(path, S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
  used += sprintf(&path[used], "%s", file_name);
  used += sprintf(&path[used], "%s", GPU_BINARY_SUFFIX);
}


uint32_t
gpu_binary_loadmap_insert
(
  const char *device_file,
  bool mark_used
)
{
  uint32_t loadmap_module_id;
  load_module_t *module = NULL;

  hpcrun_loadmap_lock();
  if ((module = hpcrun_loadmap_findByName(device_file)) == NULL) {
    loadmap_module_id = hpcrun_loadModule_add(device_file);
    module = hpcrun_loadmap_findById(loadmap_module_id);
  } else {
    loadmap_module_id = module->id;
  }
  if (mark_used) {
    hpcrun_loadModule_flags_set(module, LOADMAP_ENTRY_ANALYZE);
  }
  hpcrun_loadmap_unlock();

  return loadmap_module_id;
}


bool
gpu_binary_save
(
 const char *mem_ptr,
 size_t mem_size,
 bool mark_used,
 uint32_t *loadmap_module_id
)
{
  // Only save binaries with a valid magic number
  if (!gpu_binary_validate_magic(mem_ptr, mem_size)) return false;

  // Generate a hash for the binary
  char hash_buf[CRYPTO_HASH_STRING_LENGTH];
  crypto_compute_hash_string(mem_ptr, mem_size, hash_buf,
    CRYPTO_HASH_STRING_LENGTH);

  // Prepare to a file path to write down the binary
  char device_file[PATH_MAX];
  gpu_binary_path_generate(hash_buf, device_file);

  // Write down the binary and free the space
  bool written = gpu_binary_store(device_file, mem_ptr, mem_size);

  if (written) {
    *loadmap_module_id = gpu_binary_loadmap_insert(device_file, mark_used);
  }

  return written;
}
