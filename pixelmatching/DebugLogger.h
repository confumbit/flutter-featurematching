// reference : https://stackoverflow.com/questions/1644868/c-define-macro-for-debug-printing
#pragma once

#include <stdio.h>
#include <string>

extern void _debug_log(int level, const char *func, int line, const char *format, ...);

#define logger_e(format, ...)             \
    _debug_log(1, __func__, __LINE__, format, \
               ##__VA_ARGS__)

#define logger_i(format, ...)              \
    _debug_log(0, __func__, __LINE__, format, \
               ##__VA_ARGS__)
