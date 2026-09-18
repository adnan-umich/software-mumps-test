#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "dmumps_c.h"

#define JOB_INIT -1
#define JOB_END  -2
#define USE_COMM_WORLD -987654

static inline MUMPS_INT idx3d(int i, int j, int k, int nx, int ny)
{
    return (MUMPS_INT)(k * nx * ny + j * nx + i + 1); /* MUMPS is 1-based */
}

int main(int argc, char **argv)
{
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int nx = 64;
    int ny = 64;
    int nz = 64;

    if (argc == 4) {
        nx = atoi(argv[1]);
        ny = atoi(argv[2]);
        nz = atoi(argv[3]);
    }

    MUMPS_INT N = (MUMPS_INT)nx * ny * nz;

    /*
     * Split global rows approximately evenly across MPI ranks.
     */
    MUMPS_INT rows_per_rank = N / size;
    MUMPS_INT remainder = N % size;

    MUMPS_INT local_rows =
        rows_per_rank + (rank < remainder ? 1 : 0);

    MUMPS_INT first_row =
        rank * rows_per_rank +
        (rank < remainder ? rank : remainder);

    /* zero-based internally */
    MUMPS_INT last_row = first_row + local_rows;

    /*
     * Maximum of 7 entries per row:
     *
     *        -1
     *         |
     *    -1 --6-- -1
     *         |
     *        -1
     *
     * plus +/- z neighbors.
     */
    MUMPS_INT max_nnz = 7 * local_rows;

    MUMPS_INT *irn_loc =
        malloc((size_t)max_nnz * sizeof(MUMPS_INT));

    MUMPS_INT *jcn_loc =
        malloc((size_t)max_nnz * sizeof(MUMPS_INT));

    double *a_loc =
        malloc((size_t)max_nnz * sizeof(double));

    if (!irn_loc || !jcn_loc || !a_loc) {
        fprintf(stderr, "Rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MUMPS_INT nnz_loc = 0;

    /*
     * Build distributed 3D Laplacian.
     *
     * A has:
     *
     *     6 on diagonal
     *    -1 for each existing nearest neighbor
     */
    for (MUMPS_INT global0 = first_row;
         global0 < last_row;
         ++global0) {

        int i = global0 % nx;
        int j = (global0 / nx) % ny;
        int k = global0 / ((MUMPS_INT)nx * ny);

        MUMPS_INT row = global0 + 1;

        /* diagonal */
        irn_loc[nnz_loc] = row;
        jcn_loc[nnz_loc] = row;
        a_loc[nnz_loc]   = 6.0;
        nnz_loc++;

#define ADD_ENTRY(COL, VAL)                  \
        do {                                 \
            irn_loc[nnz_loc] = row;          \
            jcn_loc[nnz_loc] = (COL);        \
            a_loc[nnz_loc]   = (VAL);        \
            nnz_loc++;                       \
        } while (0)

        if (i > 0)
            ADD_ENTRY(idx3d(i - 1, j, k, nx, ny), -1.0);

        if (i < nx - 1)
            ADD_ENTRY(idx3d(i + 1, j, k, nx, ny), -1.0);

        if (j > 0)
            ADD_ENTRY(idx3d(i, j - 1, k, nx, ny), -1.0);

        if (j < ny - 1)
            ADD_ENTRY(idx3d(i, j + 1, k, nx, ny), -1.0);

        if (k > 0)
            ADD_ENTRY(idx3d(i, j, k - 1, nx, ny), -1.0);

        if (k < nz - 1)
            ADD_ENTRY(idx3d(i, j, k + 1, nx, ny), -1.0);

#undef ADD_ENTRY
    }

    /*
     * Centralized RHS.
     *
     * Choose exact solution:
     *
     *     x = [1, 1, ..., 1]
     *
     * Therefore:
     *
     *     b = A * 1
     *
     * For each row:
     *
     *     b_i = 6 - number_of_neighbors
     */
    double *rhs = NULL;

    if (rank == 0) {
        rhs = malloc((size_t)N * sizeof(double));

        if (!rhs) {
            fprintf(stderr, "RHS allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        for (MUMPS_INT global0 = 0; global0 < N; ++global0) {

            int i = global0 % nx;
            int j = (global0 / nx) % ny;
            int k = global0 / ((MUMPS_INT)nx * ny);

            int neighbors = 0;

            if (i > 0)      neighbors++;
            if (i < nx-1)   neighbors++;
            if (j > 0)      neighbors++;
            if (j < ny-1)   neighbors++;
            if (k > 0)      neighbors++;
            if (k < nz-1)   neighbors++;

            rhs[global0] = 6.0 - neighbors;
        }
    }

    MUMPS_INT local_nnz_tmp = nnz_loc;
    MUMPS_INT global_nnz = 0;

    MPI_Reduce(
        &local_nnz_tmp,
        &global_nnz,
        1,
        MPI_INT,
        MPI_SUM,
        0,
        MPI_COMM_WORLD
    );

    if (rank == 0) {
        printf("============================================\n");
        printf("3D Poisson MUMPS benchmark\n");
        printf("============================================\n");
        printf("Grid:       %d x %d x %d\n", nx, ny, nz);
        printf("Unknowns:   %d\n", (int)N);
        printf("Nonzeros:   %d\n", (int)global_nnz);
        printf("MPI ranks:  %d\n", size);
        printf("============================================\n");
    }

    DMUMPS_STRUC_C id;

    id.job = JOB_INIT;
    id.par = 1;
    id.sym = 0;
    id.comm_fortran = USE_COMM_WORLD;

    dmumps_c(&id);

    /*
     * Distributed assembled matrix.
     *
     * ICNTL(18) = 3
     *
     * C indexing:
     *     ICNTL(18) -> icntl[17]
     */
    id.icntl[17] = 3;

    id.n = N;

    id.nz_loc  = nnz_loc;
    id.irn_loc = irn_loc;
    id.jcn_loc = jcn_loc;
    id.a_loc   = a_loc;

    if (rank == 0)
        id.rhs = rhs;

    /*
     * Quiet-ish MUMPS output.
     */
    id.icntl[0] = -1;
    id.icntl[1] = -1;
    id.icntl[2] = -1;
    id.icntl[3] = 1;

    MPI_Barrier(MPI_COMM_WORLD);

    double t0 = MPI_Wtime();

    /*
     * Analysis + factorization + solve.
     */
    id.job = 6;
    dmumps_c(&id);

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    if (rank == 0) {

        if (id.infog[0] < 0) {

            printf("\nMUMPS FAILED\n");
            printf("INFOG(1) = %d\n", id.infog[0]);
            printf("INFOG(2) = %d\n", id.infog[1]);

        } else {

            double max_error = 0.0;

            for (MUMPS_INT i = 0; i < N; ++i) {
                double error = fabs(rhs[i] - 1.0);

                if (error > max_error)
                    max_error = error;
            }

            printf("\nSolve successful\n");
            printf("Wall time:       %.6f seconds\n", t1 - t0);
            printf("Maximum error:   %.6e\n", max_error);

            printf("\nFirst solutions:\n");

            int show = N < 10 ? N : 10;

            for (int i = 0; i < show; ++i)
                printf("x[%d] = %.12f\n", i, rhs[i]);
        }
    }

    id.job = JOB_END;
    dmumps_c(&id);

    free(irn_loc);
    free(jcn_loc);
    free(a_loc);

    if (rank == 0)
        free(rhs);

    MPI_Finalize();

    return 0;
}
