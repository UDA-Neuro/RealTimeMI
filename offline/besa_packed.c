/* BESA integer pre-compression decoder, based on the public BESA format spec. */
#include <stdint.h>
#include <stddef.h>

static int32_t signed_value(const uint8_t *p, int width) {
    uint32_t v = 0;
    for (int i = 0; i < width; ++i) v |= (uint32_t)p[i] << (8 * i);
    if (width == 1) return (int8_t)v;
    if (width == 2) return (int16_t)v;
    return (int32_t)v;
}

__declspec(dllexport) int besa_decode_packed(const uint8_t *src, size_t size,
                                              int32_t *dst, int n, int scheme,
                                              int wide_first) {
    int first_width = wide_first ? 4 : 2;
    if (n < 2 || size < (size_t)(2 * first_width)) return -1;
    dst[0] = signed_value(src, first_width);
    dst[1] = signed_value(src + first_width, first_width);
    size_t pos = (size_t)(2 * first_width);
    int k = 2;
    while (k < n) {
        if (pos >= size) return -2;
        int code = src[pos++];
        int values[4], count = 0, width = 0, raw_count = 0;
        if (scheme == 1 && code <= 224) {
            values[0] = code / 15 - 7;
            values[1] = code % 15 - 7;
            count = 2;
        } else if (scheme == 2 && code <= 124) {
            values[0] = code / 25 - 2;
            values[1] = code / 5 % 5 - 2;
            values[2] = code % 5 - 2;
            count = 3;
        } else if (scheme == 2 && code <= 245) {
            int x = code - 125;
            values[0] = x / 11 - 5;
            values[1] = x % 11 - 5;
            count = 2;
        } else if (scheme == 3 && code <= 80) {
            values[0] = code / 27 - 1;
            values[1] = code / 9 % 3 - 1;
            values[2] = code / 3 % 3 - 1;
            values[3] = code % 3 - 1;
            count = 4;
        } else if (scheme == 3 && code <= 249) {
            int x = code - 81;
            values[0] = x / 13 - 6;
            values[1] = x % 13 - 6;
            count = 2;
        } else {
            if (scheme == 1 && code >= 248 && code <= 254) {
                raw_count = 255 - code; width = 1;
            } else if (scheme == 1 && code >= 242 && code <= 247) {
                raw_count = 248 - code; width = 2;
            } else if (scheme == 1 && code >= 236 && code <= 241) {
                raw_count = 242 - code; width = 4;
            } else if (scheme == 2 && code >= 250 && code <= 254) {
                raw_count = 255 - code; width = 1;
            } else if (scheme == 2 && code >= 246 && code <= 249) {
                raw_count = 250 - code; width = 2;
            } else if (scheme == 3 && code >= 252 && code <= 254) {
                raw_count = 255 - code; width = 1;
            } else if (scheme == 3 && code >= 250 && code <= 251) {
                raw_count = 252 - code; width = 2;
            } else return -3;
            if (k + raw_count > n || pos + (size_t)(raw_count * width) > size) return -4;
            for (int i = 0; i < raw_count; ++i) {
                dst[k++] = signed_value(src + pos, width);
                pos += (size_t)width;
            }
            continue;
        }
        if (k + count > n) return -5;
        for (int i = 0; i < count; ++i) dst[k++] = values[i];
    }
    return pos == size ? 0 : -6;
}
