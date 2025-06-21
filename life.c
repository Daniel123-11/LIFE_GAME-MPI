#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIDTH 20
#define HEIGHT 20
#define STEPS 10

// Получение количества живых соседей
int count_neighbors(unsigned char* grid, int x, int y, int local_height) {
    int count = 0;
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            int nx = (x + dx + WIDTH) % WIDTH;
            int ny = (y + dy + local_height + 2) % (local_height + 2);
            count += grid[ny * WIDTH + nx];
        }
    }
    return count;
}

// Один шаг обновления
void update(unsigned char* current, unsigned char* next, int local_height) {
    for (int y = 1; y <= local_height; y++) {
        for (int x = 0; x < WIDTH; x++) {
            int neighbors = count_neighbors(current, x, y, local_height);
            int idx = y * WIDTH + x;
            if (current[idx]) {
                next[idx] = neighbors == 2 || neighbors == 3;
            } else {
                next[idx] = neighbors == 3;
            }
        }
    }
}

void print_grid(unsigned char* grid) {
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            printf("%c", grid[y * WIDTH + x] ? 'o' : '-');
        }
        printf("\n");
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_height = HEIGHT / size;
    unsigned char* local_grid = calloc((local_height + 2) * WIDTH, sizeof(unsigned char));
    unsigned char* local_next = calloc((local_height + 2) * WIDTH, sizeof(unsigned char));

    // Инициализация (только мастер)
    unsigned char* full_grid = NULL;
    if (rank == 0) {
        full_grid = calloc(HEIGHT * WIDTH, sizeof(unsigned char));
        for (int i = 0; i < HEIGHT * WIDTH; i++) {
            full_grid[i] = rand() % 2;
        }
    }

    // Рассылка подрешёток
    MPI_Scatter(full_grid, local_height * WIDTH, MPI_UNSIGNED_CHAR,
                &local_grid[WIDTH], local_height * WIDTH, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    for (int step = 0; step < STEPS; step++) {
        // Обмен границами
        int above = (rank - 1 + size) % size;
        int below = (rank + 1) % size;

        MPI_Sendrecv(&local_grid[WIDTH], WIDTH, MPI_UNSIGNED_CHAR, above, 0,
                     &local_grid[(local_height + 1) * WIDTH], WIDTH, MPI_UNSIGNED_CHAR, below, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        MPI_Sendrecv(&local_grid[local_height * WIDTH], WIDTH, MPI_UNSIGNED_CHAR, below, 1,
                     &local_grid[0], WIDTH, MPI_UNSIGNED_CHAR, above, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Обновление
        update(local_grid, local_next, local_height);

        // Обмен указателей
        unsigned char* temp = local_grid;
        local_grid = local_next;
        local_next = temp;

        // Сбор результатов на rank 0
        MPI_Gather(&local_grid[WIDTH], local_height * WIDTH, MPI_UNSIGNED_CHAR,
                   full_grid, local_height * WIDTH, MPI_UNSIGNED_CHAR,
                   0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Step %d:\n", step + 1);
            print_grid(full_grid);
            printf("\n");
        }
    }

    free(local_grid);
    free(local_next);
    if (rank == 0) free(full_grid);

    MPI_Finalize();
    return 0;
}
