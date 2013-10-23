
#ifndef _FFILE_H
#define _FFILE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>

#ifndef FFIO_MAX_ARGS
#  define FFIO_MAX_ARGS 64
#endif

struct FFILE;
typedef struct FFILE FFILE;

FFILE *
ffopen(const char *, const char *);

void
ffclose(FFILE *);

int
ffeof(FFILE *);

uint32_t
ffwrite(int, ...);

uint32_t
ffread(int, ...);

#ifdef __cplusplus
}
#endif

#endif /* _IO_H */

