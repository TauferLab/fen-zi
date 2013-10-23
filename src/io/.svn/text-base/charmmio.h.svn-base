
#ifndef _CHARMMIO_H
#define _CHARMMIO_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ffile.h"

struct charmm_trj;
typedef struct charmm_trj charmm_trj;
struct charmm_trj {
/*{{{*/
    uint8_t
        read_flag,
        first_frame;

    uint32_t
        nframes,
        current_frame_index,
        nsteps,
        initial_step_index,
        output_frequency,
        qcrys,
        qdim4,
        qcg,
        nfixed,
        ntitles,
        natoms,
        version,
        ndof,
        *freeind;

    char *titles;

    double celltmp[6];

    float
        time_step,
         *X,  *Y,  *Z,
        *X2, *Y2, *Z2,
         *C;

    FFILE *file;
};
/*}}}*/

charmm_trj *
charmm_trj_init(charmm_trj *, FFILE *);

charmm_trj *
charmm_trj_free(charmm_trj *);

void
charmm_trj_write_header(charmm_trj *);

void
charmm_trj_write_titles(charmm_trj *);

void
charmm_trj_write_natoms(charmm_trj *);

void
charmm_trj_write_freelist(charmm_trj *);

void
charmm_trj_write_frame(charmm_trj *);

#if 0 // {{{
charmm_trj *
charmm_trj_read(charmm_trj *, const char *);

charmm_trj *
charmm_trj_readf(charmm_trj *, FFILE *);

void
charmm_trj_write(charmm_trj *, const char *);

void
charmm_trj_writef(charmm_trj *, FFILE *);
#endif // }}}

#ifdef __cplusplus
}
#endif

#endif /* !_CHARMMIO_H */

