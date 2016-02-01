
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#include <io.h>

void
usage(char *argv0) {
    printf ("usage:\n"
            "\t%s [input_file]\n",
            argv0                );
}

void
charmm_trj_print(charmm_trj *ths) {
#define printint(ID) \
    printf ("%s: %ld\n", #ID, (long)(ID))

#define printfloat(ID) \
    printf ("%s: %g\n", #ID, (double)(ID))

    uint32_t i, j;

    printint( ths->nframes            );
    printint( ths->initial_step_index );
    printint( ths->output_frequency   );
    printint( ths->nfixed             );
    printint( ths->cell               );
    printint( ths->C != NULL          );

    printfloat( ths->time_step );

    printf ("NTITLE: %ld\n", ths->ntitles);

    for(i=0; i<ths->ntitles; ++i) {
        printf ("\"");
        fwrite (ths->titles+80*i, 80, 1, stdout);
        printf ("\"\n");
    }

    printf ("NFRAMES: %ld\n", ths->nframes);
    printf ("NATOMS: %ld\n" , ths->natoms );

    for(i=0; i<ths->nframes; ++i) {
        printf ("FRAME %d: \n", i+1);
        if(ths->cell) {
            printf ("\tCELL DIMENSIONS:\n");

            printf ("\t"); printfloat( ths->celltmp[i][0] );
            printf ("\t"); printfloat( ths->celltmp[i][2] );
            printf ("\t"); printfloat( ths->celltmp[i][5] );
        }

        printf ("\tCOORDINATES:\n");
        for(j=0; j<ths->natoms; ++j) {
            if(ths->C != NULL)
                printf ("\t\t% 20.16f % 20.16f % 20.16f % 20.16f\n",
                        ths->X[i][j],
                        ths->Y[i][j],
                        ths->Z[i][j],
                        ths->C[i][j]);
            else
                printf ("\t\t% 20.16f % 20.16f % 20.16f\n",
                        ths->X[i][j],
                        ths->Y[i][j],
                        ths->Z[i][j]);
        }
    }
}

int
main(int argc, char **argv) {

    charmm_trj trajectory;

    if(argc < 2) {
        usage (argv[0]);
        exit (EXIT_FAILURE);
    }

    charmm_trj_read  (&trajectory,    argv[1]);
    charmm_trj_print (&trajectory            );
    charmm_trj_write (&trajectory, "test.trj");
    charmm_trj_free  (&trajectory            );

    exit (EXIT_SUCCESS);
}

