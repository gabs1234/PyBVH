typedef unsigned int morton_t;

__device__ morton_t encodeMorton3D(int x, int y, int z) {
    morton_t answer = 0;
    for (int i = 0; i < 10; i++) {
        answer |= (x & (1 << i)) << 2 * i;
        answer |= (y & (1 << i)) << (2 * i + 1);
        answer |= (z & (1 << i)) << (2 * i + 2);
    }
    return answer;
}