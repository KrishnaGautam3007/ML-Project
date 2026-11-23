#include <string.h>
#include <stdlib.h>

void copy_input(char *src) {
    char buffer[16];
    strcpy(buffer, src);
    printf("%s\n", buffer);
}

int safe_add(int a, int b) {
    return a + b;
}

char *make_copy(const char *s, int n) {
    char *p = malloc(n);
    memcpy(p, s, n);
    return p;
}
