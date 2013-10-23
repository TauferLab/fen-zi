
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "ffile.h"

#include "charmmio.h"

static const size_t
    S_INT    = sizeof (uint32_t),
    S_FLOAT  = sizeof (float   ),
    S_DOUBLE = sizeof (double  );

charmm_trj *
charmm_trj_init(charmm_trj *ths, FFILE *f) {
/*{{{*/
    memset (ths, '\0', sizeof (*ths));
    ths->file = f;
    ths->first_frame = 1;
    return ths;
}
/*}}}*/

charmm_trj *
charmm_trj_free(charmm_trj *ths) {
/*{{{*/
    if(ths->read_flag) {
        free (ths->X);
        free (ths->Y);
        free (ths->Z);

        if(ths->C != NULL)
            free (ths->C);

        free (ths->celltmp);
        free (ths->titles);
    }

    if(ths->X2) free (ths->X2);
    if(ths->Y2) free (ths->Y2);
    if(ths->Z2) free (ths->Z2);

    return ths;
}
/*}}}*/

void
charmm_trj_write_header(charmm_trj *ths) {
/*{{{*/
    FFILE *f = ths->file;

    uint8_t buffer[80];

    uint32_t i, j, jj, nfree;

    void *ptr;

    if(ths->output_frequency == 0) {
        ths->output_frequency = 1;
    }

    ptr = buffer;
    memset (buffer, '\0', 80*sizeof (buffer[0]));
    *(uint32_t *)ptr = ths->nframes           ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->initial_step_index; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->output_frequency  ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->nsteps            ; ptr +=   S_INT  ;
                                                ptr += 3*S_INT  ;

    *(uint32_t *)ptr = ths->ndof              ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->nfixed            ; ptr +=   S_INT  ;
    *(float    *)ptr = ths->time_step         ; ptr +=   S_FLOAT;
    *(uint32_t *)ptr = ths->qcrys             ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->qdim4             ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->qcg               ; ptr +=   S_INT  ;
                                                ptr += 6*S_INT  ;
    *(uint32_t *)ptr = ths->version           ;

    ffwrite
    (2,
     "CORD",  4, sizeof (char     ),
     buffer, 80, sizeof (buffer[0]),
     f);
}
/*}}}*/

void
charmm_trj_write_titles(charmm_trj *ths) {
/*{{{*/
    FFILE *f = ths->file;

    ffwrite
    (2,
     &ths->ntitles,            1,    sizeof (ths->ntitles  ),
      ths->titles , ths->ntitles, 80*sizeof (ths->titles[0]),
     f);
}
/*}}}*/

void
charmm_trj_write_natoms(charmm_trj *ths) {
/*{{{*/
    FFILE *f = ths->file;

    ffwrite
    (1,
     &ths->natoms, 1, sizeof (ths->natoms),
     f);
}
/*}}}*/

void
charmm_trj_write_freelist(charmm_trj *ths) {
/*{{{*/
    FFILE *f = ths->file;

    uint32_t i, nfree;
    nfree = ths->natoms - ths->nfixed;

    if(ths->nfixed > 0) {
        ffwrite
        (1,
         ths->freeind, nfree, sizeof (ths->freeind[0]),
         f);
    }
}
/*}}}*/

void
charmm_trj_write_frame(charmm_trj *ths) {
/*{{{*/
    FFILE *f = ths->file;

    uint32_t j, jj, nfree;

    nfree = ths->natoms - ths->nfixed;

    if(ths->qcrys) {
        ffwrite
        (1,
         ths->celltmp, 6, sizeof (ths->celltmp[0]),
         f);
    }

    if(ths->current_frame_index==0 || ths->nfixed == 0) {
        ffwrite (1, ths->X, ths->natoms, sizeof (ths->X[0]), f);
        ffwrite (1, ths->Y, ths->natoms, sizeof (ths->Y[0]), f);
        ffwrite (1, ths->Z, ths->natoms, sizeof (ths->Z[0]), f);
    } else {
        if(!ths->X2) ths->X2 = malloc (nfree*sizeof (ths->X2[0]));
        if(!ths->Y2) ths->Y2 = malloc (nfree*sizeof (ths->Y2[0]));
        if(!ths->Z2) ths->Z2 = malloc (nfree*sizeof (ths->Z2[0]));

        for(j=0; j<nfree; ++j) {
            jj = ths->freeind[j];
            --jj;
            ths->X2[j] = ths->X[jj];
            ths->Y2[j] = ths->Y[jj];
            ths->Z2[j] = ths->Z[jj];
        }

        ffwrite (1, ths->X2, nfree, sizeof (ths->X2[0]), f);
        ffwrite (1, ths->Y2, nfree, sizeof (ths->Y2[0]), f);
        ffwrite (1, ths->Z2, nfree, sizeof (ths->Z2[0]), f);
    }

    ++ths->current_frame_index;
}
/*}}}*/


#if 0 // {{{
charmm_trj *
charmm_trj_read(charmm_trj *ths, const char *f) {
/*{{{*/
    FFILE *F;
    charmm_trj *ret;

    F = ffopen (f, "rb");
    ret = charmm_trj_readf (ths, F);
    ffclose (F);

    return ret;
}
/*}}}*/

charmm_trj *
charmm_trj_readf(charmm_trj *ths, FFILE *f) {
/*{{{*/
    uint8_t
        has_charges,
        buffer[80];

    uint32_t i, nfree, j, jj;

    char
        header[4],
        titles[32*80];

    float *X2, *Y2, *Z2;

    void *ptr;

    X2 = Y2 = Z2 = NULL;

    ffread
    (2,
     header,  4, sizeof (header[0]),
     buffer, 80, sizeof (buffer[0]),
     f);

    if(   header[0] != 'C'
       || header[1] != 'O'
       || header[2] != 'R'
       || header[3] != 'D') {

        fprintf (stderr,
                 "Error in charmm_trj_readf:\n"
                 "trajectory header mismatch\n");

        exit (EXIT_FAILURE);
    }

    ptr = buffer;

    ths->C = NULL;

    ths->nframes            = *(uint32_t *)ptr; ptr +=   S_INT  ;
    ths->initial_step_index = *(uint32_t *)ptr; ptr +=   S_INT  ;
    ths->output_frequency   = *(uint32_t *)ptr; ptr +=   S_INT  ;
                                                ptr += 5*S_INT  ;
//  ths->qdim4              = *(uint32_t *)ptr; ptr +=   S_INT  ;
    ths->nfixed             = *(uint32_t *)ptr; ptr +=   S_INT  ;
    ths->time_step          = *(float    *)ptr; ptr +=   S_FLOAT;
    ths->qcrys              = *(uint32_t *)ptr; ptr +=   S_INT  ;
                                                ptr += 8*S_INT  ;
    ths->version            = *(uint32_t *)ptr;

    ths->celltmp = malloc (ths->nframes*sizeof (ths->celltmp[0]));
    if(ths->nfixed > 0) {
        nfree = ths->natoms - ths->nfixed;
        ths->freeind = malloc (nfree*sizeof (ths->freeind[0]));
    }

    ths->X = malloc (ths->nframes*sizeof (ths->X[0]));
    ths->Y = malloc (ths->nframes*sizeof (ths->Y[0]));
    ths->Z = malloc (ths->nframes*sizeof (ths->Z[0]));
    if(has_charges) {
        ths->C = malloc (ths->nframes*sizeof (ths->C[0]));
    }

    ffread
    (2,
     &ths->ntitles, 1,                  sizeof (ths->ntitles  ),
     titles       , 0,&ths->ntitles, 80*sizeof (ths->titles[0]),
     f);

    ths->titles = malloc (ths->ntitles*80*sizeof (ths->titles[0]));
    memcpy (ths->titles, titles, ths->ntitles*80*sizeof (ths->titles[0]));

    ffread
    (1,
     &ths->natoms, 1, sizeof (ths->natoms),
     f);

    if(ths->nfixed > 0) {
        ffread
        (1,
         ths->freeind, nfree, sizeof (ths->freeind[0]),
         f);
    }

    for(i=0; i<ths->nframes; ++i) {
        if(ths->qcrys) {
            ffread
            (1,
             ths->celltmp[i], 6, sizeof (ths->celltmp[i][0]),
             f);
        }

        ths->X[i] = malloc (ths->natoms*sizeof (ths->X[i][0]));
        ths->Y[i] = malloc (ths->natoms*sizeof (ths->Y[i][0]));
        ths->Z[i] = malloc (ths->natoms*sizeof (ths->Z[i][0]));

        if(i==0 || ths->nfixed == 0) {
            ffread (1, ths->X[i], ths->natoms, sizeof (ths->X[i][0]), f);
            ffread (1, ths->Y[i], ths->natoms, sizeof (ths->Y[i][0]), f);
            ffread (1, ths->Z[i], ths->natoms, sizeof (ths->Z[i][0]), f);
        } else {
            if(!X2) X2 = malloc (nfree*sizeof (X2[0]));
            if(!Y2) Y2 = malloc (nfree*sizeof (Y2[0]));
            if(!Z2) Z2 = malloc (nfree*sizeof (Z2[0]));

            ffread (1, X2, nfree, sizeof (X2[0]), f);
            ffread (1, Y2, nfree, sizeof (Y2[0]), f);
            ffread (1, Z2, nfree, sizeof (Z2[0]), f);

            for(j=0; j<nfree; ++j) {
                jj = ths->freeind[j];
                ths->X[i][jj] = X2[j];
                ths->Y[i][jj] = Y2[j];
                ths->Z[i][jj] = Z2[j];
            }
        }

        if(has_charges) {
            ths->C[i] = malloc (ths->natoms*sizeof (*ths->C[i]));
            ffread (1, ths->C[i], ths->natoms, sizeof (*ths->C[i]), f);
        }
    }

    ths->read_flag = 1;

    if(X2) free (X2);
    if(Y2) free (Y2);
    if(Z2) free (Z2);

    return ths;
}
/*}}}*/

void
charmm_trj_write(charmm_trj *ths, const char *f) {
/*{{{*/
    FFILE *F;

    F = ffopen (f, "wb");
    charmm_trj_writef (ths, F);
    ffclose (F);
}
/*}}}*/

void
charmm_trj_writef(charmm_trj *ths, FFILE *f) {
/*{{{*/
    uint8_t
        has_charges,
        buffer[80];

    uint32_t i, j, jj, nfree;

    void *ptr;

    float *zeros = NULL;

    float *X2, *Y2, *Z2;
    X2 = Y2 = Z2 = NULL;

    nfree = ths->natoms - ths->nfixed;

    has_charges = ths->C != NULL;

    if(ths->output_frequency == 0) {
        ths->output_frequency = 1;
    }

    ptr = buffer;
    memset (buffer, '\0', 80*sizeof (buffer[0]));
    *(uint32_t *)ptr = ths->nframes           ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->initial_step_index; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->output_frequency  ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->nsteps            ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->output_frequency  ; ptr +=   S_INT  ;
                                                ptr += 2*S_INT  ;

    *(uint32_t *)ptr = ths->ndof              ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->nfixed            ; ptr +=   S_INT  ;
    *(float    *)ptr = ths->time_step         ; ptr +=   S_FLOAT;
    *(uint32_t *)ptr = ths->qcrys             ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->qdim4             ; ptr +=   S_INT  ;
    *(uint32_t *)ptr = ths->qcg               ; ptr +=   S_INT  ;
                                                ptr += 6*S_INT  ;
    *(uint32_t *)ptr = ths->version           ;

    ffwrite
    (2,
     "CORD",  4, sizeof (char     ),
     buffer, 80, sizeof (buffer[0]),
     f);

    ffwrite
    (2,
     &ths->ntitles,            1,    sizeof (ths->ntitles  ),
      ths->titles , ths->ntitles, 80*sizeof (ths->titles[0]),
     f);

    ffwrite
    (1,
     &ths->natoms, 1, sizeof (ths->natoms),
     f);

    if(ths->nfixed > 0) {
        ffwrite
        (1,
         ths->freeind, nfree, sizeof (ths->freeind[0]),
         f);
    }

    for(i=0; i<ths->nframes; ++i) {

        if(ths->qcrys) {
            ffwrite
            (1,
             ths->celltmp[i], 6, sizeof (ths->celltmp[i][0]),
             f);
        }

        if(i==0 || ths->nfixed == 0) {
            ffwrite (1, ths->X[i], ths->natoms, sizeof (*ths->X[i]), f);
            ffwrite (1, ths->Y[i], ths->natoms, sizeof (*ths->Y[i]), f);
            ffwrite (1, ths->Z[i], ths->natoms, sizeof (*ths->Z[i]), f);
        } else {
            if(!X2) X2 = malloc (nfree*sizeof (X2[0]));
            if(!Y2) Y2 = malloc (nfree*sizeof (Y2[0]));
            if(!Z2) Z2 = malloc (nfree*sizeof (Z2[0]));

            for(j=0; j<nfree; ++j) {
                jj = ths->freeind[j];
                X2[j] = ths->X[i][jj];
                Y2[j] = ths->Y[i][jj];
                Z2[j] = ths->Z[i][jj];
            }

            ffwrite (1, X2, nfree, sizeof (X2[0]), f);
            ffwrite (1, Y2, nfree, sizeof (Y2[0]), f);
            ffwrite (1, Z2, nfree, sizeof (Z2[0]), f);
        }

        /*
        if(has_charges) {
            ffwrite (1, ths->C[i], ths->natoms, sizeof (*ths->C[i]), f);
        } else {
            if(zeros == NULL) {
                zeros = calloc (ths->natoms, sizeof (zeros[0]));
            }
            ffwrite (1, zeros, ths->natoms, sizeof (zeros[0]), f);
        }
        */
    }

    if(X2) free (X2);
    if(Y2) free (Y2);
    if(Z2) free (Z2);

    /* if(zeros != NULL) free (zeros); */
}
/*}}}*/
#endif // }}}


