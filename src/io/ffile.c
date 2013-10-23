
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <stdarg.h>

#include "ffile.h"

struct FFILE {
    FILE *f;
    uint8_t first_flag, init_flag;
    uint32_t record_offset;
    uint32_t record_size;
};

/* = BEGIN = static prototypes =============================== {{{ = BEGIN = */
static inline void
ff_check_init_record(FFILE *);

static void
ff_next_record(FFILE *);

static void
ff_io_skip(FFILE *, size_t);

static void
ff_io_read(FFILE *, size_t, void *);
/* - END --- static prototypes ------------------------------- }}} --- END = */
/* = BEGIN = low-level io ==================================== {{{ = BEGIN = */
static inline void
ff_check_init_record(FFILE *ths) {
/*{{{*/
    if(ths->init_flag) {
        ths->init_flag = 0;
        ff_next_record (ths);
    }
}
/*}}}*/

static void
ff_next_record(FFILE *ths) {
/*{{{*/
    uint32_t size;
    ff_check_init_record (ths);

    if(ths->record_offset < ths->record_size) {
        fseek (ths->f,
                 ths->record_size
               - ths->record_offset,
               SEEK_CUR);

    }

    if(!ths->first_flag) {
        fread (&size, sizeof (size), 1, ths->f);

        if(ths->record_size != size ) {
            fprintf (stderr,
                       "Fortran io error: "
                       "record size header mismatch\n"
                       "%lu != %lu\n",
                     (unsigned long)size,
                     (unsigned long)ths->record_size );

            exit (EXIT_FAILURE);
        }
    }

    fread (&ths->record_size,
           sizeof (ths->record_size),
           1, ths->f               );

    ths->record_offset = 0;
    ths->first_flag    = 0;
}
/*}}}*/

static void
ff_io_skip(FFILE *ths, size_t N) {
/*{{{*/
    size_t n;
    ff_check_init_record (ths);

    while(N > 0) {
        n =   ths->record_size
            - ths->record_offset;
        if(n < N) {
            fseek (ths->f, n, SEEK_CUR);
            N -= n;
            ff_next_record (ths);
        } else {
            fseek (ths->f, N, SEEK_CUR);
            ths->record_offset += N;
            N = 0;
        }
    }
}
/*}}}*/

static void
ff_io_read(FFILE *ths, size_t N, void *out) {
/*{{{*/
    size_t n;
    ff_check_init_record (ths);

    while(N > 0) {
        n =   ths->record_size
            - ths->record_offset;
        if(n < N) {
            fread (out, 1, n, ths->f);
            N -= n;
            out += n;
            ths->record_offset += n;
            ff_next_record (ths);
        } else {
            fread (out, 1, N, ths->f);
            ths->record_offset += N;
            N = 0;
        }
    }
}
/*}}}*/
/* - END --- low-level io ------------------------------------ }}} --- END = */
FFILE *
ffopen(const char *fname, const char *mode) {
/*{{{*/
    FFILE *ret = malloc (sizeof (*ret));

    ret->f             = fopen (fname, mode);
    ret->first_flag    = 1;
    ret->init_flag     = 1;
    ret->record_offset = 0;
    ret->record_size   = 0;

    return ret;
}
/*}}}*/

void
ffclose(FFILE *ths) {
/*{{{*/
    fclose (ths->f);
    free (ths);
}
/*}}}*/

int
ffeof(FFILE *ths) {
/*{{{*/
    return feof (ths->f);
}
/*}}}*/

uint32_t
ffwrite(int nargs, ...) {
/*{{{*/
    unsigned int
        counts[FFIO_MAX_ARGS],
        sizes [FFIO_MAX_ARGS];

    uint32_t record_size;

    void *ptrs[FFIO_MAX_ARGS];
    int i;

    va_list L;
    FFILE *ths;

    if(nargs > FFIO_MAX_ARGS) {
        fprintf (stderr,
                   "Error at ffwrite(): "
                   "too many arguments (%d/%d)\n",
                 nargs, FFIO_MAX_ARGS          );

        exit (EXIT_FAILURE);
    }

    record_size = 0;
    va_start( L, nargs );
    for(i=0; i<nargs; ++i) {
        ptrs  [i] = va_arg( L, void *       );
        sizes [i] = va_arg( L, unsigned int );
        counts[i] = va_arg( L, unsigned int );

        record_size += sizes[i]*counts[i];
    }
    ths = va_arg( L, FFILE * );
    va_end( L );

    fwrite (&record_size, 1, sizeof (record_size), ths->f);
    for(i=0; i<nargs; ++i) {
        if(ptrs[i] == NULL) {
            fseek (ths->f, sizes[i]*counts[i], SEEK_CUR);
        } else {
            fwrite (ptrs[i], counts[i], sizes[i], ths->f);
        }
    }
    fwrite (&record_size, 1, sizeof (record_size), ths->f);

    return record_size;
}
/*}}}*/

uint32_t
ffread(int nargs, ...) {
/*{{{*/
    unsigned int
        size, count,
        counts[FFIO_MAX_ARGS],
        sizes[FFIO_MAX_ARGS],
        *size_ptrs[FFIO_MAX_ARGS],
        *count_ptrs[FFIO_MAX_ARGS];

    uint32_t
        read_size,
        record_size;

    void *ptrs[FFIO_MAX_ARGS];
    int i;

    va_list L;
    FFILE *ths;

    if(nargs > FFIO_MAX_ARGS) {
        fprintf (stderr,
                   "Error at ffread(): "
                   "too many arguments (%d/%d)\n",
                 nargs, FFIO_MAX_ARGS          );

        exit (EXIT_FAILURE);
    }

    va_start( L, nargs );
    for(i=0; i<nargs; ++i) {
        count_ptrs[i] = NULL;
        size_ptrs[i] = NULL;

        ptrs[i] = va_arg( L, void * );

        counts[i] = va_arg( L, unsigned int );
        if(counts[i] == 0) {
            count_ptrs[i] = va_arg( L, unsigned int * );
        }

        sizes[i] = va_arg( L, unsigned int );
        if(sizes[i] == 0) {
            size_ptrs[i] = va_arg( L, unsigned int * );
        }
    }
    ths = va_arg( L, FFILE * );
    va_end( L );

    read_size = 0;

    for(i=0; i<nargs; ++i) {
        count = counts[i];
        size  =  sizes[i];

        if(count == 0) count = *count_ptrs[i];
        if(size  == 0) size  =  *size_ptrs[i];

        read_size += count*size;

        if(ffeof (ths)) {
            fprintf (stderr,
                       "Error at ffread(): "
                       "EOF reached.\n"    );

            exit (EXIT_FAILURE);
        }

        if(ptrs[i] == NULL) {
            ff_io_skip (ths, count*size);
        } else {
            ff_io_read (ths, count*size, ptrs[i]);
        }

    }

    ff_next_record (ths);

    return read_size;
}
/*}}}*/

