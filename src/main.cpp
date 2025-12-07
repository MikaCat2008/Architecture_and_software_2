#include <stdio.h>
#include <intrin.h>
#include <iostream>
#include <time.h>
#include <wchar.h>
#include <stdio.h>
#include <locale.h>
#include <stdlib.h>
#include <io.h>
#include <fcntl.h>


static const int PANEL_main_menu = 0;
static const int PANEL_cpuid = 1;
static const int PANEL_input_variant = 2;
static const int PANEL_input_type = 3;
static const int PANEL_input_n = 4;
static const int PANEL_mult = 5;
static const int PANEL_settings = 6;

static const char* TEXT_variants[10] = {
    "a * (A * x) + b * (B * y)",
    "A * x + b * (B * y)",
    "a * (A * x) + B * y",
    "a * (A * x) + y",
    "a * x + b * (B * y)",
    "a * (A * x) - b * (B * y)",
    "A * x - b * (B * y)",
    "a * (A * x) - B * y",
    "a * (A * x) - y",
    "a * x - b * (B * y)"
};
static const char* TEXT_types[3] = {
    "int", "float", "double"
};

static bool SETTING_save_input = false;
static bool SETTING_save_output = false;
static bool SETTING_save_result = false;
static bool SETTING_target = false;

static int op_variant, op_type, op_n;


template<typename... Args>
int winputd(const wchar_t *text, Args... args) {
    int value;

    wprintf(text, args...);
    if (wscanf(L"%d", &value) == 0) {
        wint_t c;
        while ((c = getwchar()) != L'\n' && c != WEOF);

        throw L"Ви ввели не число!";
    }
    
    return value;
}

template<typename... Args>
double winputf(const wchar_t *text, Args... args) {
    double value;

    wprintf(text, args...);
    if (wscanf(L"%lf", &value) == 0) {
        wint_t c;
        while ((c = getwchar()) != L'\n' && c != WEOF);

        throw 0;
    }

    return value;
}

// --- v1


void v1_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) + b * (B[i * n + j] * y[j]);
}

void v1_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i sex_b = _mm_set1_epi32(b);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum_A = _mm_setzero_si128();
            __m128i sex_sum_B = _mm_setzero_si128();

            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_epi32(sex_sum_A, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_epi32(sex_sum_B, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_hadd_epi32(sex_sum_A, sex_sum_A);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_A, sex_sum_A));

            sex_sum_B = _mm_hadd_epi32(sex_sum_B, sex_sum_B);
            sum[si] += b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_B, sex_sum_B));
        }

        sex_r[i] = _mm_add_epi32(sex_r[i], _mm_loadu_si128((__m128i*)sum));
    }
}

void v1_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i ssex_b = _mm256_set1_epi32(b);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum_A = _mm256_setzero_si256();
            __m256i ssex_sum_B = _mm256_setzero_si256();

            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_epi32(
                    ssex_sum_A, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j])
                );
                ssex_sum_B = _mm256_add_epi32(
                    ssex_sum_B, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j])
                );
            }

            __m128i sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_A),
                _mm256_extracti128_si256(ssex_sum_A, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));

            sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_B),
                _mm256_extracti128_si256(ssex_sum_B, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] += b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(ssex_r[i], _mm256_loadu_si256((__m256i*)sum));
    }
}


void v1_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) + b * (B[i * n + j] * y[j]);
}

void v1_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128 sex_b = _mm_set1_ps(b);
    __m128* sex_A = (__m128*)A;
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum_A = _mm_setzero_ps();
            __m128 sex_sum_B = _mm_setzero_ps();

            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_ps(sex_sum_A, _mm_mul_ps(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_ps(sex_sum_B, _mm_mul_ps(sex_Bi[j], sex_y[j]));
            }

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum_A);
            sex_sum_A = _mm_add_ps(sex_sum_A, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum_A, _mm_movehl_ps(sex_shuf, sex_sum_A)));

            sex_shuf = _mm_movehdup_ps(sex_sum_B);
            sex_sum_B = _mm_add_ps(sex_sum_B, sex_shuf);
            sum[si] += b * _mm_cvtss_f32(_mm_add_ss(sex_sum_B, _mm_movehl_ps(sex_shuf, sex_sum_B)));
        }

        sex_r[i] = _mm_add_ps(sex_r[i], _mm_loadu_ps(sum));
    }
}

void v1_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256 ssex_b = _mm256_set1_ps(b);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum_A = _mm256_setzero_ps();
            __m256 ssex_sum_B = _mm256_setzero_ps();

            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_ps(ssex_sum_A, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_ps(ssex_sum_B, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            }

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_A),
                _mm256_extractf128_ps(ssex_sum_A, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));

            sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_B),
                _mm256_extractf128_ps(ssex_sum_B, 1)
            );
            sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] += b * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(ssex_r[i], _mm256_loadu_ps(sum));
    }
}


void v1_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) + b * (B[i * n + j] * y[j]);
}

void v1_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d sex_b = _mm_set1_pd(b);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum_A = _mm_setzero_pd();
            __m128d sex_sum_B = _mm_setzero_pd();

            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++) {
                sex_sum_A = _mm_add_pd(sex_sum_A, _mm_mul_pd(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_pd(sex_sum_B, _mm_mul_pd(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_add_pd(sex_sum_A, _mm_shuffle_pd(sex_sum_A, sex_sum_A, 0x1));
            sum[si] = a * _mm_cvtsd_f64(sex_sum_A);

            sex_sum_B = _mm_add_pd(sex_sum_B, _mm_shuffle_pd(sex_sum_B, sex_sum_B, 0x1));
            sum[si] += b * _mm_cvtsd_f64(sex_sum_B);
        }

        sex_r[i] = _mm_add_pd(sex_r[i], _mm_loadu_pd(sum));
    }
}

void v1_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d ssex_b = _mm256_set1_pd(b);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum_A = _mm256_setzero_pd();
            __m256d ssex_sum_B = _mm256_setzero_pd();

            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                ssex_sum_A = _mm256_add_pd(ssex_sum_A, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_pd(ssex_sum_B, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));
            }

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_A),
                _mm256_extractf128_pd(ssex_sum_A, 1)
            );
            sum[si] = a * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));

            sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_B),
                _mm256_extractf128_pd(ssex_sum_B, 1)
            );
            sum[si] += b * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(ssex_r[i], _mm256_loadu_pd(sum));
    }
}


void call_v1_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v1_base_int(n, a, b, A, B, x, y, r);
}

void call_v1_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v1_sse_int(n, a, b, A, B, x, y, r);
}

void call_v1_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v1_avx_int(n, a, b, A, B, x, y, r);
}


void call_v1_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v1_base_float(n, a, b, A, B, x, y, r);
}

void call_v1_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v1_sse_float(n, a, b, A, B, x, y, r);
}

void call_v1_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v1_avx_float(n, a, b, A, B, x, y, r);
}


void call_v1_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v1_base_double(n, a, b, A, B, x, y, r);
}

void call_v1_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v1_sse_double(n, a, b, A, B, x, y, r);
}

void call_v1_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v1_avx_double(n, a, b, A, B, x, y, r);
}


// --- v1
// --- v2


void v2_base_int(int n, int b, int* A, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j] + b * (B[i * n + j] * y[j]);
}

void v2_sse_int(int n, int b, int* A, int* B, int* x, int* y, int* r) {
    __m128i sex_b = _mm_set1_epi32(b);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum_A = _mm_setzero_si128();
            __m128i sex_sum_B = _mm_setzero_si128();

            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_epi32(sex_sum_A, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_epi32(sex_sum_B, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_hadd_epi32(sex_sum_A, sex_sum_A);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_A, sex_sum_A));

            sex_sum_B = _mm_hadd_epi32(sex_sum_B, sex_sum_B);
            sum[si] += b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_B, sex_sum_B));
        }

        sex_r[i] = _mm_add_epi32(sex_r[i], _mm_loadu_si128((__m128i*)sum));
    }
}

void v2_avx_int(int n, int b, int* A, int* B, int* x, int* y, int* r) {
    __m256i ssex_b = _mm256_set1_epi32(b);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum_A = _mm256_setzero_si256();
            __m256i ssex_sum_B = _mm256_setzero_si256();

            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_epi32(
                    ssex_sum_A, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j])
                );
                ssex_sum_B = _mm256_add_epi32(
                    ssex_sum_B, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j])
                );
            }

            __m128i sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_A),
                _mm256_extracti128_si256(ssex_sum_A, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));

            sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_B),
                _mm256_extracti128_si256(ssex_sum_B, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] += b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(ssex_r[i], _mm256_loadu_si256((__m256i*)sum));
    }
}


void v2_base_float(int n, float b, float* A, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j] + b * (B[i * n + j] * y[j]);
}

void v2_sse_float(int n, float b, float* A, float* B, float* x, float* y, float* r) {
    __m128 sex_b = _mm_set1_ps(b);
    __m128* sex_A = (__m128*)A;
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum_A = _mm_setzero_ps();
            __m128 sex_sum_B = _mm_setzero_ps();

            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_ps(sex_sum_A, _mm_mul_ps(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_ps(sex_sum_B, _mm_mul_ps(sex_Bi[j], sex_y[j]));
            }

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum_A);
            sex_sum_A = _mm_add_ps(sex_sum_A, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum_A, _mm_movehl_ps(sex_shuf, sex_sum_A)));

            sex_shuf = _mm_movehdup_ps(sex_sum_B);
            sex_sum_B = _mm_add_ps(sex_sum_B, sex_shuf);
            sum[si] += b * _mm_cvtss_f32(_mm_add_ss(sex_sum_B, _mm_movehl_ps(sex_shuf, sex_sum_B)));
        }

        sex_r[i] = _mm_add_ps(sex_r[i], _mm_loadu_ps(sum));
    }
}

void v2_avx_float(int n, float b, float* A, float* B, float* x, float* y, float* r) {
    __m256 ssex_b = _mm256_set1_ps(b);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum_A = _mm256_setzero_ps();
            __m256 ssex_sum_B = _mm256_setzero_ps();

            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_ps(ssex_sum_A, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_ps(ssex_sum_B, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            }

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_A),
                _mm256_extractf128_ps(ssex_sum_A, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));

            sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_B),
                _mm256_extractf128_ps(ssex_sum_B, 1)
            );
            sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] += b * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(ssex_r[i], _mm256_loadu_ps(sum));
    }
}


void v2_base_double(int n, double b, double* A, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j] + b * (B[i * n + j] * y[j]);
}

void v2_sse_double(int n, double b, double* A, double* B, double* x, double* y, double* r) {
    __m128d sex_b = _mm_set1_pd(b);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum_A = _mm_setzero_pd();
            __m128d sex_sum_B = _mm_setzero_pd();

            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++) {
                sex_sum_A = _mm_add_pd(sex_sum_A, _mm_mul_pd(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_pd(sex_sum_B, _mm_mul_pd(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_add_pd(sex_sum_A, _mm_shuffle_pd(sex_sum_A, sex_sum_A, 0x1));
            sum[si] = _mm_cvtsd_f64(sex_sum_A);

            sex_sum_B = _mm_add_pd(sex_sum_B, _mm_shuffle_pd(sex_sum_B, sex_sum_B, 0x1));
            sum[si] += b * _mm_cvtsd_f64(sex_sum_B);
        }

        sex_r[i] = _mm_add_pd(sex_r[i], _mm_loadu_pd(sum));
    }
}

void v2_avx_double(int n, double b, double* A, double* B, double* x, double* y, double* r) {
    __m256d ssex_b = _mm256_set1_pd(b);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum_A = _mm256_setzero_pd();
            __m256d ssex_sum_B = _mm256_setzero_pd();

            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                ssex_sum_A = _mm256_add_pd(ssex_sum_A, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_pd(ssex_sum_B, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));
            }

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_A),
                _mm256_extractf128_pd(ssex_sum_A, 1)
            );
            sum[si] = _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));

            sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_B),
                _mm256_extractf128_pd(ssex_sum_B, 1)
            );
            sum[si] += b * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(ssex_r[i], _mm256_loadu_pd(sum));
    }
}


void call_v2_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v2_base_int(n, b, A, B, x, y, r);
}

void call_v2_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v2_sse_int(n, b, A, B, x, y, r);
}

void call_v2_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v2_avx_int(n, b, A, B, x, y, r);
}


void call_v2_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v2_base_float(n, b, A, B, x, y, r);
}

void call_v2_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v2_sse_float(n, b, A, B, x, y, r);
}

void call_v2_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v2_avx_float(n, b, A, B, x, y, r);
}


void call_v2_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v2_base_double(n, b, A, B, x, y, r);
}

void call_v2_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v2_sse_double(n, b, A, B, x, y, r);
}

void call_v2_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v2_avx_double(n, b, A, B, x, y, r);
}


// --- v2
// --- v3


void v3_base_int(int n, int a, int* A, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) + (B[i * n + j] * y[j]);
}

void v3_sse_int(int n, int a, int* A, int* B, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum_A = _mm_setzero_si128();
            __m128i sex_sum_B = _mm_setzero_si128();

            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_epi32(sex_sum_A, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_epi32(sex_sum_B, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_hadd_epi32(sex_sum_A, sex_sum_A);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_A, sex_sum_A));

            sex_sum_B = _mm_hadd_epi32(sex_sum_B, sex_sum_B);
            sum[si] += _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_B, sex_sum_B));
        }

        sex_r[i] = _mm_add_epi32(sex_r[i], _mm_loadu_si128((__m128i*)sum));
    }
}

void v3_avx_int(int n, int a, int* A, int* B, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum_A = _mm256_setzero_si256();
            __m256i ssex_sum_B = _mm256_setzero_si256();

            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_epi32(
                    ssex_sum_A, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j])
                );
                ssex_sum_B = _mm256_add_epi32(
                    ssex_sum_B, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j])
                );
            }

            __m128i sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_A), 
                _mm256_extracti128_si256(ssex_sum_A, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));

            sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_B), 
                _mm256_extracti128_si256(ssex_sum_B, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] += _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(ssex_r[i], _mm256_loadu_si256((__m256i*)sum));
    }
}


void v3_base_float(int n, float a, float* A, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) + B[i * n + j] * y[j];
}

void v3_sse_float(int n, float a, float* A, float* B, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128* sex_A = (__m128*)A;
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum_A = _mm_setzero_ps();
            __m128 sex_sum_B = _mm_setzero_ps();

            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_ps(sex_sum_A, _mm_mul_ps(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_ps(sex_sum_B, _mm_mul_ps(sex_Bi[j], sex_y[j]));
            }

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum_A);
            sex_sum_A = _mm_add_ps(sex_sum_A, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum_A, _mm_movehl_ps(sex_shuf, sex_sum_A)));

            sex_shuf = _mm_movehdup_ps(sex_sum_B);
            sex_sum_B = _mm_add_ps(sex_sum_B, sex_shuf);
            sum[si] += _mm_cvtss_f32(_mm_add_ss(sex_sum_B, _mm_movehl_ps(sex_shuf, sex_sum_B)));
        }

        __m128 res = _mm_loadu_ps(sum);
        sex_r[i] = _mm_add_ps(sex_r[i], res);
    }
}

void v3_avx_float(int n, float a, float* A, float* B, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum_A = _mm256_setzero_ps();
            __m256 ssex_sum_B = _mm256_setzero_ps();

            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_ps(ssex_sum_A, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_ps(ssex_sum_B, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            }

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_A), 
                _mm256_extractf128_ps(ssex_sum_A, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));

            sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_B), 
                _mm256_extractf128_ps(ssex_sum_B, 1)
            );
            sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] += _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(ssex_r[i], _mm256_loadu_ps(sum));
    }
}


void v3_base_double(int n, double a, double* A, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) + B[i * n + j] * y[j];
}

void v3_sse_double(int n, double a, double* A, double* B, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum_A = _mm_setzero_pd();
            __m128d sex_sum_B = _mm_setzero_pd();

            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++) {
                sex_sum_A = _mm_add_pd(sex_sum_A, _mm_mul_pd(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_pd(sex_sum_B, _mm_mul_pd(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_add_pd(sex_sum_A, _mm_shuffle_pd(sex_sum_A, sex_sum_A, 0x1));
            sum[si]  = a * _mm_cvtsd_f64(sex_sum_A);

            sex_sum_B = _mm_add_pd(sex_sum_B, _mm_shuffle_pd(sex_sum_B, sex_sum_B, 0x1));
            sum[si] += _mm_cvtsd_f64(sex_sum_B);
        }

        sex_r[i] = _mm_add_pd(sex_r[i], _mm_loadu_pd(sum));
    }
}

void v3_avx_double(int n, double a, double* A, double* B, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum_A = _mm256_setzero_pd();
            __m256d ssex_sum_B = _mm256_setzero_pd();

            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                ssex_sum_A = _mm256_add_pd(ssex_sum_A, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_pd(ssex_sum_B, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));
            }

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_A),
                _mm256_extractf128_pd(ssex_sum_A, 1)
            );
            sum[si] = a * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));

            sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_B),
                _mm256_extractf128_pd(ssex_sum_B, 1)
            );
            sum[si] += _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(ssex_r[i], _mm256_loadu_pd(sum));
    }
}


void call_v3_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v3_base_int(n, a, A, B, x, y, r);
}

void call_v3_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v3_sse_int(n, a, A, B, x, y, r);
}

void call_v3_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v3_avx_int(n, a, A, B, x, y, r);
}


void call_v3_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v3_base_float(n, a, A, B, x, y, r);
}

void call_v3_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v3_sse_float(n, a, A, B, x, y, r);
}

void call_v3_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v3_avx_float(n, a, A, B, x, y, r);
}


void call_v3_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v3_base_double(n, a, A, B, x, y, r);
}

void call_v3_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v3_sse_double(n, a, A, B, x, y, r);
}

void call_v3_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v3_avx_double(n, a, A, B, x, y, r);
}


// --- v3
// --- v4


void v4_base_int(int n, int a, int* A, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = y[i];

        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]);
    }
}

void v4_sse_int(int n, int a, int* A, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum = _mm_setzero_si128();
            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_epi32(sex_sum, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));

            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        sex_r[i] = _mm_add_epi32(
            _mm_mullo_epi32(sex_a, _mm_loadu_si128((__m128i*)sum)),
            sex_y[i]
        );
    }
}

void v4_avx_int(int n, int a, int* A, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum = _mm256_setzero_si256();
            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_epi32(
                    ssex_sum, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j])
                );

            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum), 
                _mm256_extracti128_si256(ssex_sum, 1)
            );

            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sum128, sum128));
        }

        ssex_r[i] = _mm256_add_epi32(
            _mm256_mullo_epi32(ssex_a, _mm256_loadu_si256((__m256i*)sum)),
            ssex_y[i]
        );
    }
}


void v4_base_float(int n, float a, float* A, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = y[i];

        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]);
    }
}

void v4_sse_float(int n, float a, float* A, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128* sex_A = (__m128*)A;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum = _mm_setzero_ps();
            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_ps(sex_sum, _mm_mul_ps(sex_Ai[j], sex_x[j]));

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);

            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        sex_r[i] = _mm_add_ps(
            _mm_mul_ps(sex_a, _mm_loadu_ps(sum)), 
            sex_y[i]
        );
    }
}

void v4_avx_float(int n, float a, float* A, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum = _mm256_setzero_ps();
            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_ps(ssex_sum, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum), 
                _mm256_extractf128_ps(ssex_sum, 1)
            );

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);

            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(
            _mm256_mul_ps(ssex_a, _mm256_loadu_ps(sum)), 
            ssex_y[i]
        );
    }
}


void v4_base_double(int n, double a, double* A, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = y[i];

        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]);
    }
}

void v4_sse_double(int n, double a, double* A, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum = _mm_setzero_pd();
            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++)
                sex_sum = _mm_add_pd(sex_sum, _mm_mul_pd(sex_Ai[j], sex_x[j]));

            sum[si] = _mm_cvtsd_f64(
                _mm_add_sd(sex_sum, _mm_shuffle_pd(sex_sum, sex_sum, 0x1))
            );
        }

        sex_r[i] = _mm_add_pd(
            _mm_mul_pd(sex_a, _mm_loadu_pd(sum)),
            sex_y[i]
        );
    }
}

void v4_avx_double(int n, double a, double* A, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum = _mm256_setzero_pd();
            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                ssex_sum = _mm256_add_pd(ssex_sum, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum), 
                _mm256_extractf128_pd(ssex_sum, 1)
            );

            sum[si] = _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(
            _mm256_mul_pd(ssex_a, _mm256_loadu_pd(sum)),
            ssex_y[i]
        );
    }
}


void call_v4_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v4_base_int(n, a, A, x, y, r);
}

void call_v4_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v4_sse_int(n, a, A, x, y, r);
}

void call_v4_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v4_avx_int(n, a, A, x, y, r);
}


void call_v4_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v4_base_float(n, a, A, x, y, r);
}

void call_v4_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v4_sse_float(n, a, A, x, y, r);
}

void call_v4_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v4_avx_float(n, a, A, x, y, r);
}


void call_v4_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v4_base_double(n, a, A, x, y, r);
}

void call_v4_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v4_sse_double(n, a, A, x, y, r);
}

void call_v4_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v4_avx_double(n, a, A, x, y, r);
}


// --- v4
// --- v5


void v5_base_int(int n, int a, int b, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = a * x[i];

        for (size_t j = 0; j < n; j++)
            r[i] += b * (B[i * n + j] * y[j]);
    }
}

void v5_sse_int(int n, int a, int b, int* B, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i sex_b = _mm_set1_epi32(b);
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;

    for (int i = 0; i < n4; i++) {
        int sum[4];

        for (int si = 0; si < 4; si++) {
            __m128i sex_sum = _mm_setzero_si128();
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_epi32(sex_sum, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));

            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));      
        }

        sex_r[i] = _mm_add_epi32(
            _mm_mullo_epi32(sex_a, sex_x[i]),
            _mm_mullo_epi32(sex_b, _mm_loadu_si128((__m128i*)sum))
        );
    }
}

void v5_avx_int(int n, int a, int b, int* B, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i ssex_b = _mm256_set1_epi32(b);
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum = _mm256_setzero_si256();
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_epi32(
                    ssex_sum, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j])
                );

            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum), 
                _mm256_extracti128_si256(ssex_sum, 1)
            );

            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sum128, sum128));
        }

        ssex_r[i] = _mm256_add_epi32(
            _mm256_mullo_epi32(ssex_a, ssex_x[i]),
            _mm256_mullo_epi32(ssex_b, _mm256_loadu_si256((__m256i*)sum))
        );
    }
}


void v5_base_float(int n, float a, float b, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = a * x[i];

        for (size_t j = 0; j < n; j++)
            r[i] += b * (B[i * n + j] * y[j]);
    }
}

void v5_sse_float(int n, float a, float b, float* B, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128 sex_b = _mm_set1_ps(b);
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum = _mm_setzero_ps();
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_ps(sex_sum, _mm_mul_ps(sex_Bi[j], sex_y[j]));

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        sex_r[i] = _mm_add_ps(
            _mm_mul_ps(sex_a, sex_x[i]), 
            _mm_mul_ps(sex_b, _mm_loadu_ps(sum))
        );
    }
}

void v5_avx_float(int n, float a, float b, float* B, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256 ssex_b = _mm256_set1_ps(b);
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum = _mm256_setzero_ps();
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_ps(ssex_sum, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            
            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum), 
                _mm256_extractf128_ps(ssex_sum, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(
            _mm256_mul_ps(ssex_a, ssex_x[i]),
            _mm256_mul_ps(ssex_b, _mm256_loadu_ps(sum))
        );
    }
}


void v5_base_double(int n, double a, double b, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = a * x[i];

        for (size_t j = 0; j < n; j++)
            r[i] += b * (B[i * n + j] * y[j]);
    }
}

void v5_sse_double(int n, double a, double b, double* B, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d sex_b = _mm_set1_pd(b);
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum = _mm_setzero_pd();
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++)
                sex_sum = _mm_add_pd(sex_sum, _mm_mul_pd(sex_Bi[j], sex_y[j]));

            __m128d sex_shuf = _mm_shuffle_pd(sex_sum, sex_sum, 0x1);
            sum[si] = _mm_cvtsd_f64(_mm_add_sd(sex_sum, sex_shuf));
        }

        sex_r[i] = _mm_add_pd(
            _mm_mul_pd(sex_a, sex_x[i]),
            _mm_mul_pd(sex_b, _mm_loadu_pd(sum))
        );
    }
}

void v5_avx_double(int n, double a, double b, double* B, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d ssex_b = _mm256_set1_pd(b);
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum = _mm256_setzero_pd();
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                ssex_sum = _mm256_add_pd(ssex_sum, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum), 
                _mm256_extractf128_pd(ssex_sum, 1)
            );           
            sum[si] = _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));                  
        }

        ssex_r[i] = _mm256_add_pd(
            _mm256_mul_pd(ssex_a, ssex_x[i]),
            _mm256_mul_pd(ssex_b, _mm256_loadu_pd(sum))
        );
    }
}


void call_v5_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v5_base_int(n, a, b, B, x, y, r);
}

void call_v5_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v5_sse_int(n, a, b, B, x, y, r);
}

void call_v5_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v5_avx_int(n, a, b, B, x, y, r);
}


void call_v5_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v5_base_float(n, a, b, B, x, y, r);
}

void call_v5_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v5_sse_float(n, a, b, B, x, y, r);
}

void call_v5_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v5_avx_float(n, a, b, B, x, y, r);
}


void call_v5_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v5_base_double(n, a, b, B, x, y, r);
}

void call_v5_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v5_sse_double(n, a, b, B, x, y, r);
}

void call_v5_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v5_avx_double(n, a, b, B, x, y, r);
}


// --- v5
// --- v6


void v6_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) - b * (B[i * n + j] * y[j]);
}

void v6_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i sex_b = _mm_set1_epi32(b);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum_A = _mm_setzero_si128();
            __m128i sex_sum_B = _mm_setzero_si128();

            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_epi32(sex_sum_A, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_epi32(sex_sum_B, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_hadd_epi32(sex_sum_A, sex_sum_A);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_A, sex_sum_A));

            sex_sum_B = _mm_hadd_epi32(sex_sum_B, sex_sum_B);
            sum[si] -= b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_B, sex_sum_B));
        }

        sex_r[i] = _mm_add_epi32(sex_r[i], _mm_loadu_si128((__m128i*)sum));
    }
}

void v6_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i ssex_b = _mm256_set1_epi32(b);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum_A = _mm256_setzero_si256();
            __m256i ssex_sum_B = _mm256_setzero_si256();

            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_epi32(
                    ssex_sum_A, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j])
                );
                ssex_sum_B = _mm256_add_epi32(
                    ssex_sum_B, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j])
                );
            }

            __m128i sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_A),
                _mm256_extracti128_si256(ssex_sum_A, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));

            sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_B),
                _mm256_extracti128_si256(ssex_sum_B, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] -= b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(ssex_r[i], _mm256_loadu_si256((__m256i*)sum));
    }
}


void v6_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) - b * (B[i * n + j] * y[j]);
}

void v6_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128 sex_b = _mm_set1_ps(b);
    __m128* sex_A = (__m128*)A;
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum_A = _mm_setzero_ps();
            __m128 sex_sum_B = _mm_setzero_ps();

            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_ps(sex_sum_A, _mm_mul_ps(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_ps(sex_sum_B, _mm_mul_ps(sex_Bi[j], sex_y[j]));
            }

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum_A);
            sex_sum_A = _mm_add_ps(sex_sum_A, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum_A, _mm_movehl_ps(sex_shuf, sex_sum_A)));

            sex_shuf = _mm_movehdup_ps(sex_sum_B);
            sex_sum_B = _mm_add_ps(sex_sum_B, sex_shuf);
            sum[si] -= b * _mm_cvtss_f32(_mm_add_ss(sex_sum_B, _mm_movehl_ps(sex_shuf, sex_sum_B)));
        }

        sex_r[i] = _mm_add_ps(sex_r[i], _mm_loadu_ps(sum));
    }
}

void v6_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256 ssex_b = _mm256_set1_ps(b);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum_A = _mm256_setzero_ps();
            __m256 ssex_sum_B = _mm256_setzero_ps();

            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_ps(ssex_sum_A, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_ps(ssex_sum_B, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            }

            __m128 sex_sum = _mm_add_ps(_mm256_castps256_ps128(ssex_sum_A), _mm256_extractf128_ps(ssex_sum_A, 1));
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));

            sex_sum = _mm_add_ps(_mm256_castps256_ps128(ssex_sum_B), _mm256_extractf128_ps(ssex_sum_B, 1));
            sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] -= b * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(ssex_r[i], _mm256_loadu_ps(sum));
    }
}


void v6_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) - b * (B[i * n + j] * y[j]);
}

void v6_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d sex_b = _mm_set1_pd(b);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum_A = _mm_setzero_pd();
            __m128d sex_sum_B = _mm_setzero_pd();

            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++) {
                sex_sum_A = _mm_add_pd(sex_sum_A, _mm_mul_pd(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_pd(sex_sum_B, _mm_mul_pd(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_add_pd(sex_sum_A, _mm_shuffle_pd(sex_sum_A, sex_sum_A, 0x1));
            sum[si] = a * _mm_cvtsd_f64(sex_sum_A);

            sex_sum_B = _mm_add_pd(sex_sum_B, _mm_shuffle_pd(sex_sum_B, sex_sum_B, 0x1));
            sum[si] -= b * _mm_cvtsd_f64(sex_sum_B);
        }

        sex_r[i] = _mm_add_pd(sex_r[i], _mm_loadu_pd(sum));
    }
}

void v6_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d ssex_b = _mm256_set1_pd(b);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum_A = _mm256_setzero_pd();
            __m256d ssex_sum_B = _mm256_setzero_pd();

            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                ssex_sum_A = _mm256_add_pd(ssex_sum_A, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_pd(ssex_sum_B, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));
            }

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_A),
                _mm256_extractf128_pd(ssex_sum_A, 1)
            );
            sum[si] = a * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));

            sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_B),
                _mm256_extractf128_pd(ssex_sum_B, 1)
            );
            sum[si] -= b * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(ssex_r[i], _mm256_loadu_pd(sum));
    }
}


void call_v6_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v6_base_int(n, a, b, A, B, x, y, r);
}

void call_v6_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v6_sse_int(n, a, b, A, B, x, y, r);
}

void call_v6_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v6_avx_int(n, a, b, A, B, x, y, r);
}


void call_v6_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v6_base_float(n, a, b, A, B, x, y, r);
}

void call_v6_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v6_sse_float(n, a, b, A, B, x, y, r);
}

void call_v6_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v6_avx_float(n, a, b, A, B, x, y, r);
}


void call_v6_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v6_base_double(n, a, b, A, B, x, y, r);
}

void call_v6_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v6_sse_double(n, a, b, A, B, x, y, r);
}

void call_v6_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v6_avx_double(n, a, b, A, B, x, y, r);
}


// --- v6
// --- v7


void v7_base_int(int n, int b, int* A, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j] - b * (B[i * n + j] * y[j]);
}

void v7_sse_int(int n, int b, int* A, int* B, int* x, int* y, int* r) {
    __m128i sex_b = _mm_set1_epi32(b);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum_A = _mm_setzero_si128();
            __m128i sex_sum_B = _mm_setzero_si128();

            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_epi32(sex_sum_A, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_epi32(sex_sum_B, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_hadd_epi32(sex_sum_A, sex_sum_A);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_A, sex_sum_A));

            sex_sum_B = _mm_hadd_epi32(sex_sum_B, sex_sum_B);
            sum[si] -= b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_B, sex_sum_B));
        }

        sex_r[i] = _mm_add_epi32(sex_r[i], _mm_loadu_si128((__m128i*)sum));
    }
}

void v7_avx_int(int n, int b, int* A, int* B, int* x, int* y, int* r) {
    __m256i ssex_b = _mm256_set1_epi32(b);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum_A = _mm256_setzero_si256();
            __m256i ssex_sum_B = _mm256_setzero_si256();

            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_epi32(ssex_sum_A, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_epi32(ssex_sum_B, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j]));
            }

            __m128i sex_sum = _mm_add_epi32(_mm256_castsi256_si128(ssex_sum_A), _mm256_extracti128_si256(ssex_sum_A, 1));
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));

            sex_sum = _mm_add_epi32(_mm256_castsi256_si128(ssex_sum_B), _mm256_extracti128_si256(ssex_sum_B, 1));
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] -= b * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(ssex_r[i], _mm256_loadu_si256((__m256i*)sum));
    }
}


void v7_base_float(int n, float b, float* A, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j] - b * (B[i * n + j] * y[j]);
}

void v7_sse_float(int n, float b, float* A, float* B, float* x, float* y, float* r) {
    __m128 sex_b = _mm_set1_ps(b);
    __m128* sex_A = (__m128*)A;
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum_A = _mm_setzero_ps();
            __m128 sex_sum_B = _mm_setzero_ps();

            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_ps(sex_sum_A, _mm_mul_ps(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_ps(sex_sum_B, _mm_mul_ps(sex_Bi[j], sex_y[j]));
            }

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum_A);
            sex_sum_A = _mm_add_ps(sex_sum_A, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum_A, _mm_movehl_ps(sex_shuf, sex_sum_A)));

            sex_shuf = _mm_movehdup_ps(sex_sum_B);
            sex_sum_B = _mm_add_ps(sex_sum_B, sex_shuf);
            sum[si] -= b * _mm_cvtss_f32(_mm_add_ss(sex_sum_B, _mm_movehl_ps(sex_shuf, sex_sum_B)));
        }

        sex_r[i] = _mm_add_ps(sex_r[i], _mm_loadu_ps(sum));
    }
}

void v7_avx_float(int n, float b, float* A, float* B, float* x, float* y, float* r) {
    __m256 ssex_b = _mm256_set1_ps(b);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum_A = _mm256_setzero_ps();
            __m256 ssex_sum_B = _mm256_setzero_ps();

            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_ps(ssex_sum_A, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_ps(ssex_sum_B, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            }

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_A), 
                _mm256_extractf128_ps(ssex_sum_A, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));

            sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_B), 
                _mm256_extractf128_ps(ssex_sum_B, 1)
            );
            sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] -= b * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(ssex_r[i], _mm256_loadu_ps(sum));
    }
}


void v7_base_double(int n, double b, double* A, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += A[i * n + j] * x[j] - b * (B[i * n + j] * y[j]);
}

void v7_sse_double(int n, double b, double* A, double* B, double* x, double* y, double* r) {
    __m128d sex_b = _mm_set1_pd(b);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum_A = _mm_setzero_pd();
            __m128d sex_sum_B = _mm_setzero_pd();

            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++) {
                sex_sum_A = _mm_add_pd(sex_sum_A, _mm_mul_pd(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_pd(sex_sum_B, _mm_mul_pd(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_add_pd(sex_sum_A, _mm_shuffle_pd(sex_sum_A, sex_sum_A, 0x1));
            sum[si] = _mm_cvtsd_f64(sex_sum_A);

            sex_sum_B = _mm_add_pd(sex_sum_B, _mm_shuffle_pd(sex_sum_B, sex_sum_B, 0x1));
            sum[si] -= b * _mm_cvtsd_f64(sex_sum_B);
        }

        sex_r[i] = _mm_add_pd(sex_r[i], _mm_loadu_pd(sum));
    }
}

void v7_avx_double(int n, double b, double* A, double* B, double* x, double* y, double* r) {
    __m256d ssex_b = _mm256_set1_pd(b);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum_A = _mm256_setzero_pd();
            __m256d ssex_sum_B = _mm256_setzero_pd();

            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                ssex_sum_A = _mm256_add_pd(ssex_sum_A, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_pd(ssex_sum_B, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));
            }

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_A),
                _mm256_extractf128_pd(ssex_sum_A, 1)
            );
            sum[si] = _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));

            sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_B),
                _mm256_extractf128_pd(ssex_sum_B, 1)
            );
            sum[si] -= b * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(ssex_r[i], _mm256_loadu_pd(sum));
    }
}


void call_v7_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v7_base_int(n, b, A, B, x, y, r);
}

void call_v7_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v7_sse_int(n, b, A, B, x, y, r);
}

void call_v7_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v7_avx_int(n, b, A, B, x, y, r);
}


void call_v7_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v7_base_float(n, b, A, B, x, y, r);
}

void call_v7_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v7_sse_float(n, b, A, B, x, y, r);
}

void call_v7_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v7_avx_float(n, b, A, B, x, y, r);
}


void call_v7_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v7_base_double(n, b, A, B, x, y, r);
}

void call_v7_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v7_sse_double(n, b, A, B, x, y, r);
}

void call_v7_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v7_avx_double(n, b, A, B, x, y, r);
}


// --- v7
// --- v8


void v8_base_int(int n, int a, int* A, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) - (B[i * n + j] * y[j]);
}

void v8_sse_int(int n, int a, int* A, int* B, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum_A = _mm_setzero_si128();
            __m128i sex_sum_B = _mm_setzero_si128();

            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_epi32(sex_sum_A, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_epi32(sex_sum_B, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_hadd_epi32(sex_sum_A, sex_sum_A);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_A, sex_sum_A));

            sex_sum_B = _mm_hadd_epi32(sex_sum_B, sex_sum_B);
            sum[si] -= _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum_B, sex_sum_B));
        }

        sex_r[i] = _mm_add_epi32(sex_r[i], _mm_loadu_si128((__m128i*)sum));
    }
}

void v8_avx_int(int n, int a, int* A, int* B, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum_A = _mm256_setzero_si256();
            __m256i ssex_sum_B = _mm256_setzero_si256();

            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_epi32(ssex_sum_A, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_epi32(ssex_sum_B, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j]));
            }

            __m128i sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_A),
                _mm256_extracti128_si256(ssex_sum_A, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = a * _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));

            sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum_B),
                _mm256_extracti128_si256(ssex_sum_B, 1)
            );
            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] -= _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(ssex_r[i], _mm256_loadu_si256((__m256i*)sum));
    }
}


void v8_base_float(int n, float a, float* A, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) - B[i * n + j] * y[j];
}

void v8_sse_float(int n, float a, float* A, float* B, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128* sex_A = (__m128*)A;
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum_A = _mm_setzero_ps();
            __m128 sex_sum_B = _mm_setzero_ps();

            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                sex_sum_A = _mm_add_ps(sex_sum_A, _mm_mul_ps(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_ps(sex_sum_B, _mm_mul_ps(sex_Bi[j], sex_y[j]));
            }

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum_A);
            sex_sum_A = _mm_add_ps(sex_sum_A, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum_A, _mm_movehl_ps(sex_shuf, sex_sum_A)));

            sex_shuf = _mm_movehdup_ps(sex_sum_B);
            sex_sum_B = _mm_add_ps(sex_sum_B, sex_shuf);
            sum[si] -= _mm_cvtss_f32(_mm_add_ss(sex_sum_B, _mm_movehl_ps(sex_shuf, sex_sum_B)));
        }

        sex_r[i] = _mm_add_ps(sex_r[i], _mm_loadu_ps(sum));
    }
}

void v8_avx_float(int n, float a, float* A, float* B, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum_A = _mm256_setzero_ps();
            __m256 ssex_sum_B = _mm256_setzero_ps();

            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++) {
                ssex_sum_A = _mm256_add_ps(ssex_sum_A, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_ps(ssex_sum_B, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));
            }

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_A),
                _mm256_extractf128_ps(ssex_sum_A, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = a * _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));

            sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum_B),
                _mm256_extractf128_ps(ssex_sum_B, 1)
            );
            sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] -= _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(ssex_r[i], _mm256_loadu_ps(sum));
    }
}


void v8_base_double(int n, double a, double* A, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++)
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]) - B[i * n + j] * y[j];
}

void v8_sse_double(int n, double a, double* A, double* B, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum_A = _mm_setzero_pd();
            __m128d sex_sum_B = _mm_setzero_pd();

            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++) {
                sex_sum_A = _mm_add_pd(sex_sum_A, _mm_mul_pd(sex_Ai[j], sex_x[j]));
                sex_sum_B = _mm_add_pd(sex_sum_B, _mm_mul_pd(sex_Bi[j], sex_y[j]));
            }

            sex_sum_A = _mm_add_pd(sex_sum_A, _mm_shuffle_pd(sex_sum_A, sex_sum_A, 0x1));
            sum[si] = a * _mm_cvtsd_f64(sex_sum_A);

            sex_sum_B = _mm_add_pd(sex_sum_B, _mm_shuffle_pd(sex_sum_B, sex_sum_B, 0x1));
            sum[si] -= _mm_cvtsd_f64(sex_sum_B);
        }

        sex_r[i] = _mm_add_pd(sex_r[i], _mm_loadu_pd(sum));
    }
}

void v8_avx_double(int n, double a, double* A, double* B, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum_A = _mm256_setzero_pd();
            __m256d ssex_sum_B = _mm256_setzero_pd();

            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++) {
                ssex_sum_A = _mm256_add_pd(ssex_sum_A, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));
                ssex_sum_B = _mm256_add_pd(ssex_sum_B, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));
            }

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_A),
                _mm256_extractf128_pd(ssex_sum_A, 1)
            );
            sum[si] = a * _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));

            sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum_B),
                _mm256_extractf128_pd(ssex_sum_B, 1)
            );
            sum[si] -= _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(ssex_r[i], _mm256_loadu_pd(sum));
    }
}


void call_v8_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v8_base_int(n, a, A, B, x, y, r);
}

void call_v8_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v8_sse_int(n, a, A, B, x, y, r);
}

void call_v8_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v8_avx_int(n, a, A, B, x, y, r);
}


void call_v8_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v8_base_float(n, a, A, B, x, y, r);
}

void call_v8_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v8_sse_float(n, a, A, B, x, y, r);
}

void call_v8_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v8_avx_float(n, a, A, B, x, y, r);
}


void call_v8_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v8_base_double(n, a, A, B, x, y, r);
}

void call_v8_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v8_sse_double(n, a, A, B, x, y, r);
}

void call_v8_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v8_avx_double(n, a, A, B, x, y, r);
}


// --- v8
// --- v9


void v9_base_int(int n, int a, int* A, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = -y[i];

        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]);
    }
}

void v9_sse_int(int n, int a, int* A, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i* sex_A = (__m128i*)A;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;
    int sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128i sex_sum = _mm_setzero_si128();
            __m128i* sex_Ai = sex_A + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_epi32(sex_sum, _mm_mullo_epi32(sex_Ai[j], sex_x[j]));

            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        sex_r[i] = _mm_sub_epi32(
            _mm_mullo_epi32(sex_a, _mm_loadu_si128((__m128i*)sum)),
            sex_y[i]
        );
    }
}

void v9_avx_int(int n, int a, int* A, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i* ssex_A = (__m256i*)A;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum = _mm256_setzero_si256();
            __m256i* ssex_Ai = ssex_A + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_epi32(
                    ssex_sum, _mm256_mullo_epi32(ssex_Ai[j], ssex_x[j])
                );

            __m128i sum128 = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum), 
                _mm256_extracti128_si256(ssex_sum, 1)
            );

            sum128 = _mm_hadd_epi32(sum128, sum128);
            sum[si] = _mm_cvtsi128_si32(_mm_hadd_epi32(sum128, sum128));
        }

        ssex_r[i] = _mm256_sub_epi32(
            _mm256_mullo_epi32(ssex_a, _mm256_loadu_si256((__m256i*)sum)),
            ssex_y[i]
        );
    }
}


void v9_base_float(int n, float a, float* A, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = -y[i];
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]);
    }
}

void v9_sse_float(int n, float a, float* A, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128* sex_A = (__m128*)A;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum = _mm_setzero_ps();
            __m128* sex_Ai = sex_A + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_ps(sex_sum, _mm_mul_ps(sex_Ai[j], sex_x[j]));

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        sex_r[i] = _mm_add_ps(
            _mm_mul_ps(sex_a, _mm_loadu_ps(sum)),
            _mm_sub_ps(_mm_setzero_ps(), sex_y[i])
        );
    }
}

void v9_avx_float(int n, float a, float* A, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256* ssex_A = (__m256*)A;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum = _mm256_setzero_ps();
            __m256* ssex_Ai = ssex_A + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_ps(ssex_sum, _mm256_mul_ps(ssex_Ai[j], ssex_x[j]));

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum),
                _mm256_extractf128_ps(ssex_sum, 1)
            );

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = _mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(
            _mm256_mul_ps(ssex_a, _mm256_loadu_ps(sum)),
            _mm256_sub_ps(_mm256_setzero_ps(), ssex_y[i])
        );
    }
}


void v9_base_double(int n, double a, double* A, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = -y[i];
        for (size_t j = 0; j < n; j++)
            r[i] += a * (A[i * n + j] * x[j]);
    }
}

void v9_sse_double(int n, double a, double* A, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d* sex_A = (__m128d*)A;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum = _mm_setzero_pd();
            __m128d* sex_Ai = sex_A + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++)
                sex_sum = _mm_add_pd(sex_sum, _mm_mul_pd(sex_Ai[j], sex_x[j]));

            sum[si] = _mm_cvtsd_f64(_mm_add_sd(sex_sum, _mm_shuffle_pd(sex_sum, sex_sum, 0x1)));
        }

        sex_r[i] = _mm_add_pd(
            _mm_mul_pd(sex_a, _mm_loadu_pd(sum)),
            _mm_sub_pd(_mm_setzero_pd(), sex_y[i])
        );
    }
}

void v9_avx_double(int n, double a, double* A, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d* ssex_A = (__m256d*)A;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum = _mm256_setzero_pd();
            __m256d* ssex_Ai = ssex_A + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                ssex_sum = _mm256_add_pd(ssex_sum, _mm256_mul_pd(ssex_Ai[j], ssex_x[j]));

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum),
                _mm256_extractf128_pd(ssex_sum, 1)
            );

            sum[si] = _mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(
            _mm256_mul_pd(ssex_a, _mm256_loadu_pd(sum)),
            _mm256_sub_pd(_mm256_setzero_pd(), ssex_y[i])
        );
    }
}


void call_v9_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v9_base_int(n, a, A, x, y, r);
}

void call_v9_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v9_sse_int(n, a, A, x, y, r);
}

void call_v9_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v9_avx_int(n, a, A, x, y, r);
}


void call_v9_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v9_base_float(n, a, A, x, y, r);
}

void call_v9_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v9_sse_float(n, a, A, x, y, r);
}

void call_v9_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v9_avx_float(n, a, A, x, y, r);
}


void call_v9_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v9_base_double(n, a, A, x, y, r);
}

void call_v9_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v9_sse_double(n, a, A, x, y, r);
}

void call_v9_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v9_avx_double(n, a, A, x, y, r);
}


// --- v9
// --- v10


void v10_base_int(int n, int a, int b, int* B, int* x, int* y, int* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = a * x[i];
        for (size_t j = 0; j < n; j++)
            r[i] -= b * (B[i * n + j] * y[j]);
    }
}

void v10_sse_int(int n, int a, int b, int* B, int* x, int* y, int* r) {
    __m128i sex_a = _mm_set1_epi32(a);
    __m128i sex_b = _mm_set1_epi32(b);
    __m128i* sex_B = (__m128i*)B;
    __m128i* sex_x = (__m128i*)x;
    __m128i* sex_y = (__m128i*)y;
    __m128i* sex_r = (__m128i*)r;

    int n4 = n / 4;

    for (int i = 0; i < n4; i++) {
        int sum[4];

        for (int si = 0; si < 4; si++) {
            __m128i sex_sum = _mm_setzero_si128();
            __m128i* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_epi32(sex_sum, _mm_mullo_epi32(sex_Bi[j], sex_y[j]));

            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = -_mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        sex_r[i] = _mm_add_epi32(
            _mm_mullo_epi32(sex_a, sex_x[i]),
            _mm_mullo_epi32(sex_b, _mm_loadu_si128((__m128i*)sum))
        );
    }
}

void v10_avx_int(int n, int a, int b, int* B, int* x, int* y, int* r) {
    __m256i ssex_a = _mm256_set1_epi32(a);
    __m256i ssex_b = _mm256_set1_epi32(b);
    __m256i* ssex_B = (__m256i*)B;
    __m256i* ssex_x = (__m256i*)x;
    __m256i* ssex_y = (__m256i*)y;
    __m256i* ssex_r = (__m256i*)r;

    int n8 = n / 8;
    int sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256i ssex_sum = _mm256_setzero_si256();
            __m256i* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_epi32(
                    ssex_sum, _mm256_mullo_epi32(ssex_Bi[j], ssex_y[j])
                );

            __m128i sex_sum = _mm_add_epi32(
                _mm256_castsi256_si128(ssex_sum), 
                _mm256_extracti128_si256(ssex_sum, 1)
            );

            sex_sum = _mm_hadd_epi32(sex_sum, sex_sum);
            sum[si] = -_mm_cvtsi128_si32(_mm_hadd_epi32(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_epi32(
            _mm256_mullo_epi32(ssex_a, ssex_x[i]),
            _mm256_mullo_epi32(ssex_b, _mm256_loadu_si256((__m256i*)sum))
        );
    }
}


void v10_base_float(int n, float a, float b, float* B, float* x, float* y, float* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = a * x[i];
        for (size_t j = 0; j < n; j++)
            r[i] -= b * (B[i * n + j] * y[j]);
    }
}

void v10_sse_float(int n, float a, float b, float* B, float* x, float* y, float* r) {
    __m128 sex_a = _mm_set1_ps(a);
    __m128 sex_b = _mm_set1_ps(b);
    __m128* sex_B = (__m128*)B;
    __m128* sex_x = (__m128*)x;
    __m128* sex_y = (__m128*)y;
    __m128* sex_r = (__m128*)r;

    int n4 = n / 4;
    float sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m128 sex_sum = _mm_setzero_ps();
            __m128* sex_Bi = sex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                sex_sum = _mm_add_ps(sex_sum, _mm_mul_ps(sex_Bi[j], sex_y[j]));

            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = -_mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        sex_r[i] = _mm_add_ps(_mm_mul_ps(sex_a, sex_x[i]), _mm_mul_ps(sex_b, _mm_loadu_ps(sum)));
    }
}

void v10_avx_float(int n, float a, float b, float* B, float* x, float* y, float* r) {
    __m256 ssex_a = _mm256_set1_ps(a);
    __m256 ssex_b = _mm256_set1_ps(b);
    __m256* ssex_B = (__m256*)B;
    __m256* ssex_x = (__m256*)x;
    __m256* ssex_y = (__m256*)y;
    __m256* ssex_r = (__m256*)r;

    int n8 = n / 8;
    float sum[8];

    for (int i = 0; i < n8; i++) {
        for (int si = 0; si < 8; si++) {
            __m256 ssex_sum = _mm256_setzero_ps();
            __m256* ssex_Bi = ssex_B + (i * 8 + si) * n8;

            for (int j = 0; j < n8; j++)
                ssex_sum = _mm256_add_ps(ssex_sum, _mm256_mul_ps(ssex_Bi[j], ssex_y[j]));

            __m128 sex_sum = _mm_add_ps(
                _mm256_castps256_ps128(ssex_sum), 
                _mm256_extractf128_ps(ssex_sum, 1)
            );
            __m128 sex_shuf = _mm_movehdup_ps(sex_sum);
            sex_sum = _mm_add_ps(sex_sum, sex_shuf);
            sum[si] = -_mm_cvtss_f32(_mm_add_ss(sex_sum, _mm_movehl_ps(sex_shuf, sex_sum)));
        }

        ssex_r[i] = _mm256_add_ps(_mm256_mul_ps(ssex_a, ssex_x[i]), _mm256_mul_ps(ssex_b, _mm256_loadu_ps(sum)));
    }
}


void v10_base_double(int n, double a, double b, double* B, double* x, double* y, double* r) {
    for (size_t i = 0; i < n; i++) {
        r[i] = a * x[i];
        for (size_t j = 0; j < n; j++)
            r[i] -= b * (B[i * n + j] * y[j]);
    }
}

void v10_sse_double(int n, double a, double b, double* B, double* x, double* y, double* r) {
    __m128d sex_a = _mm_set1_pd(a);
    __m128d sex_b = _mm_set1_pd(b);
    __m128d* sex_B = (__m128d*)B;
    __m128d* sex_x = (__m128d*)x;
    __m128d* sex_y = (__m128d*)y;
    __m128d* sex_r = (__m128d*)r;

    int n2 = n / 2;
    double sum[2];

    for (int i = 0; i < n2; i++) {
        for (int si = 0; si < 2; si++) {
            __m128d sex_sum = _mm_setzero_pd();
            __m128d* sex_Bi = sex_B + (i * 2 + si) * n2;

            for (int j = 0; j < n2; j++)
                sex_sum = _mm_add_pd(sex_sum, _mm_mul_pd(sex_Bi[j], sex_y[j]));

            __m128d sex_shuf = _mm_shuffle_pd(sex_sum, sex_sum, 0x1);
            sum[si] = -_mm_cvtsd_f64(_mm_add_sd(sex_sum, sex_shuf));
        }

        sex_r[i] = _mm_add_pd(_mm_mul_pd(sex_a, sex_x[i]), _mm_mul_pd(sex_b, _mm_loadu_pd(sum)));
    }
}

void v10_avx_double(int n, double a, double b, double* B, double* x, double* y, double* r) {
    __m256d ssex_a = _mm256_set1_pd(a);
    __m256d ssex_b = _mm256_set1_pd(b);
    __m256d* ssex_B = (__m256d*)B;
    __m256d* ssex_x = (__m256d*)x;
    __m256d* ssex_y = (__m256d*)y;
    __m256d* ssex_r = (__m256d*)r;

    int n4 = n / 4;
    double sum[4];

    for (int i = 0; i < n4; i++) {
        for (int si = 0; si < 4; si++) {
            __m256d ssex_sum = _mm256_setzero_pd();
            __m256d* ssex_Bi = ssex_B + (i * 4 + si) * n4;

            for (int j = 0; j < n4; j++)
                ssex_sum = _mm256_add_pd(ssex_sum, _mm256_mul_pd(ssex_Bi[j], ssex_y[j]));

            __m128d sex_sum = _mm_add_pd(
                _mm256_castpd256_pd128(ssex_sum), 
                _mm256_extractf128_pd(ssex_sum, 1)
            );
            sum[si] = -_mm_cvtsd_f64(_mm_hadd_pd(sex_sum, sex_sum));
        }

        ssex_r[i] = _mm256_add_pd(_mm256_mul_pd(ssex_a, ssex_x[i]), _mm256_mul_pd(ssex_b, _mm256_loadu_pd(sum)));
    }
}


void call_v10_base_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v10_base_int(n, a, b, B, x, y, r);
}

void call_v10_sse_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v10_sse_int(n, a, b, B, x, y, r);
}

void call_v10_avx_int(int n, int a, int b, int* A, int* B, int* x, int* y, int* r) {
    v10_avx_int(n, a, b, B, x, y, r);
}


void call_v10_base_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v10_base_float(n, a, b, B, x, y, r);
}

void call_v10_sse_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v10_sse_float(n, a, b, B, x, y, r);
}

void call_v10_avx_float(int n, float a, float b, float* A, float* B, float* x, float* y, float* r) {
    v10_avx_float(n, a, b, B, x, y, r);
}


void call_v10_base_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v10_base_double(n, a, b, B, x, y, r);
}

void call_v10_sse_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v10_sse_double(n, a, b, B, x, y, r);
}

void call_v10_avx_double(int n, double a, double b, double* A, double* B, double* x, double* y, double* r) {
    v10_avx_double(n, a, b, B, x, y, r);
}


// --- v10


void print_main_menu() {
    wprintf(
        L"Головне меню\n"
        L"1. Підтримка SSE/AVX\n"
        L"2. Варіанти\n"
        L"3. Налаштування\n"
        L"4. Вихід\n"
    );
}

void print_cpuid() {
    bool sse, sse_2, sse_3, sse_41, sse_42, sse_4a, sse_5, avx;

	int cpuinfo[4];
	__cpuid(cpuinfo, 1);

	sse = cpuinfo[3] & (1 << 25) || false;
	sse_2 = cpuinfo[3] & (1 << 26) || false;
	sse_3 = cpuinfo[2] & (1 << 0) || false;
	sse_41 = cpuinfo[2] & (1 << 19) || false;
	sse_42 = cpuinfo[2] & (1 << 20) || false;

	avx = cpuinfo[2] & (1 << 28) || false;
	if ((cpuinfo[2] & (1 << 27) || false) && avx)
		avx = (_xgetbv(0) & 0x6) == 0x6;

	__cpuid(cpuinfo, 0x80000000);
	if (cpuinfo[0] >= 0x80000001)
	{
		__cpuid(cpuinfo, 0x80000001);
		sse_4a = cpuinfo[2] & (1 << 6) || false;
		sse_5 = cpuinfo[2] & (1 << 11) || false;
	}

    wprintf(
        L"Цей пк підтримує такі SSE/AVX інструкції:\n"
	    L"SSE    : %s\n"
	    L"SSE2   : %s\n"
	    L"SSE3   : %s\n"
	    L"SSE4.1 : %s\n"
	    L"SSE4.2 : %s\n"
	    L"SSE4a  : %s\n"
	    L"SSE5   : %s\n"
	    L"AVX    : %s\n"
        L"1. Назад\n",
        sse    ? "+" : "-",
        sse_2  ? "+" : "-",
        sse_3  ? "+" : "-",
        sse_41 ? "+" : "-",
        sse_42 ? "+" : "-",
        sse_4a ? "+" : "-",
        sse_5  ? "+" : "-",
        avx    ? "+" : "-"
    );
}

void print_input_variant() {
    wprintf(
        L"Варіанти завдань множення матриць з векторами та скалярами:\n"
        L" 1. Операція aAx + bBy\n"
        L" 2. Операція  Ax + bBy\n"
        L" 3. Операція aAx +  By\n"
        L" 4. Операція aAx +   y\n"
        L" 5. Операція a x + bBy\n"
        L" 6. Операція aAx - bBy\n"
        L" 7. Операція  Ax - bBy\n"
        L" 8. Операція aAx -  By\n"
        L" 9. Операція aAx -   y\n"
        L"10. Операція a x - bBy\n"
        L"11. Назад\n"
    );
}

void print_input_type() {
    wprintf(
        L"Оберіть тип даних, для якого буде проводитись виконання операції:\n"
        L"1. int\n"
        L"2. float\n"
        L"3. double\n"
        L"4. Назад\n"
    );
}

void print_input_n() {
    wprintf(
        L"Напишіть розмір матриці (кратний 8), або\n"
        L"1. Назад\n"
    );
}

template <typename T>
void measure_time(
    void (*call_base)(int, T, T, T*, T*, T*, T*, T*), 
    void (*call_sse)(int, T, T, T*, T*, T*, T*, T*), 
    void (*call_avx)(int, T, T, T*, T*, T*, T*, T*)
) {
    char filepath[200] = "results\\result-";

    T a = rand() % 200 - 100;
    T b = rand() % 200 - 100;
    T* A = (T*)_mm_malloc(op_n * op_n * sizeof(T), 32);
    T* B = (T*)_mm_malloc(op_n * op_n * sizeof(T), 32);
    T* x = (T*)_mm_malloc(op_n * sizeof(T), 32);
    T* y = (T*)_mm_malloc(op_n * sizeof(T), 32);
    T* r = (T*)_mm_malloc(op_n * sizeof(T), 32);

    for (size_t i = 0; i < op_n * op_n; i++) {
        A[i] = rand() % 200 - 100;
        B[i] = rand() % 200 - 100;
    }

    for (size_t i = 0; i < op_n; i++) {
        x[i] = rand() % 200 - 100;
        y[i] = rand() % 200 - 100;
    }

    if (SETTING_save_result) {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);

        char buffer[100];
        strftime(buffer, sizeof(buffer), "%d.%m.%Y-%H-%M-%S", t);

        strcat(filepath, buffer);
        strcat(filepath, ".txt");

        FILE *fp = fopen(filepath, "a");

        fprintf(fp, "Операція: %s\n", TEXT_variants[op_variant - 1]);
        fprintf(fp, "Тип: %s\n", TEXT_types[op_type - 1]);
        fprintf(fp, "N=%d\n", op_n);

        if (SETTING_save_input) {
            if constexpr (std::is_same_v<T, int>)
                fprintf(fp, "a=%4d\n", a);
            else
                fprintf(fp, "a=%6.1f\n", a);

            if constexpr (std::is_same_v<T, int>)
                fprintf(fp, "b=%4d\n", b);
            else
                fprintf(fp, "b=%6.1f\n", b);

            fprintf(fp, "A=");

            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++) {            
                    fprintf(fp, "\n  ");

                    for (size_t j = 0; j < op_n; j++)
                        fprintf(fp, "%4d ", A[i * op_n + j]);            
                }
            else
                for (size_t i = 0; i < op_n; i++) {            
                    fprintf(fp, "\n  ");

                    for (size_t j = 0; j < op_n; j++)
                        fprintf(fp, "%6.1f ", A[i * op_n + j]);            
                }

            fprintf(fp, "\nB=");

            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++) {            
                    fprintf(fp, "\n  ");

                    for (size_t j = 0; j < op_n; j++)
                        fprintf(fp, "%4d ", B[i * op_n + j]);            
                }
            else
                for (size_t i = 0; i < op_n; i++) {            
                    fprintf(fp, "\n  ");

                    for (size_t j = 0; j < op_n; j++)
                        fprintf(fp, "%6.1f ", B[i * op_n + j]);            
                }

            fprintf(fp, "\nx=\n  ");

            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%4d ", x[i]);
            else 
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%6.1f ", x[i]);

            fprintf(fp, "\ny=\n  ");

            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%4d ", y[i]);
            else
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%6.1f ", y[i]);

            fprintf(fp, "\n");
        }

        fclose(fp);
    }

    for (size_t i = 0; i < op_n; i++)
        r[i] = 0;

    clock_t start = clock();

    call_base(op_n, a, b, A, B, x, y, r);

    double s = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    wprintf(L"Час виконання: %f секунд\n", s);

    if (SETTING_save_result) {
        FILE *fp = fopen(filepath, "a");
        
        if (SETTING_save_output) {
            fprintf(fp, "\nresult base=\n  ");
        
            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%8d ", r[i]);
            else
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%10.1f ", r[i]);
        }

        fprintf(fp, "\nЧас виконання: %f секунд\n", s);

        fclose(fp);
    }

    for (size_t i = 0; i < op_n; i++)
        r[i] = 0;
    
    start = clock();

    call_sse(op_n, a, b, A, B, x, y, r);

    s = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    wprintf(L"Час виконання SSE: %f секунд\n", s);

    if (SETTING_save_result) {
        FILE *fp = fopen(filepath, "a");

        if (SETTING_save_output) {
            fprintf(fp, "\nresult sse=\n  ");
            
            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%8d ", r[i]);
            else
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%10.1f ", r[i]);
        }

        fprintf(fp, "\nЧас виконання SSE: %f секунд\n", s);

        fclose(fp);
    }

    for (size_t i = 0; i < op_n; i++)
        r[i] = 0;
    
    start = clock();

    call_avx(op_n, a, b, A, B, x, y, r);

    s = ((double)(clock() - start)) / CLOCKS_PER_SEC;
    wprintf(L"Час виконання AVX: %f секунд\n", s);

    if (SETTING_save_result) {
        FILE *fp = fopen(filepath, "a");

        if (SETTING_save_output) {
            fprintf(fp, "\nresult avx=\n  ");

            if constexpr (std::is_same_v<T, int>)
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%8d ", r[i]);
            else
                for (size_t i = 0; i < op_n; i++)
                    fprintf(fp, "%10.1f ", r[i]);
        }

        fprintf(fp, "\nЧас виконання AVX: %f секунд\n", s);
        wprintf(L"Файл: \"%hs\"\n", filepath);

        fclose(fp);
    }

    _mm_free(A);
    _mm_free(B);
    _mm_free(x);
    _mm_free(y);
    _mm_free(r);
}


void process_variant() {
    switch (op_variant)
    {
    case 1: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v1_base_int, 
                call_v1_sse_int, 
                call_v1_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v1_base_float, 
                call_v1_sse_float, 
                call_v1_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v1_base_double, 
                call_v1_sse_double, 
                call_v1_avx_double
            );

            break;
        };

        break;
    }

    case 2: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v2_base_int, 
                call_v2_sse_int, 
                call_v2_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v2_base_float, 
                call_v2_sse_float, 
                call_v2_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v2_base_double, 
                call_v2_sse_double, 
                call_v2_avx_double
            );

            break;
        };

        break;
    }

    case 3: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v3_base_int, 
                call_v3_sse_int, 
                call_v3_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v3_base_float, 
                call_v3_sse_float, 
                call_v3_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v3_base_double, 
                call_v3_sse_double, 
                call_v3_avx_double
            );

            break;
        };

        break;
    }

    case 4: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v4_base_int, 
                call_v4_sse_int, 
                call_v4_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v4_base_float, 
                call_v4_sse_float, 
                call_v4_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v4_base_double, 
                call_v4_sse_double, 
                call_v4_avx_double
            );

            break;
        };

        break;
    }

    case 5: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v5_base_int, 
                call_v5_sse_int, 
                call_v5_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v5_base_float, 
                call_v5_sse_float, 
                call_v5_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v5_base_double, 
                call_v5_sse_double, 
                call_v5_avx_double
            );

            break;
        };

        break;
    }

    case 6: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v6_base_int, 
                call_v6_sse_int, 
                call_v6_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v6_base_float, 
                call_v6_sse_float, 
                call_v6_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v6_base_double, 
                call_v6_sse_double, 
                call_v6_avx_double
            );

            break;
        };

        break;
    }

    case 7: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v7_base_int, 
                call_v7_sse_int, 
                call_v7_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v7_base_float, 
                call_v7_sse_float, 
                call_v7_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v7_base_double, 
                call_v7_sse_double, 
                call_v7_avx_double
            );

            break;
        };

        break;
    }

    case 8: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v8_base_int, 
                call_v8_sse_int, 
                call_v8_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v8_base_float, 
                call_v8_sse_float, 
                call_v8_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v8_base_double, 
                call_v8_sse_double, 
                call_v8_avx_double
            );

            break;
        };

        break;
    }

    case 9: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v9_base_int, 
                call_v9_sse_int, 
                call_v9_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v9_base_float, 
                call_v9_sse_float, 
                call_v9_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v9_base_double, 
                call_v9_sse_double, 
                call_v9_avx_double
            );

            break;
        };

        break;
    }

    case 10: {
        switch (op_type)
        {
        case 1:
            measure_time<int>(
                call_v10_base_int, 
                call_v10_sse_int, 
                call_v10_avx_int
            );

            break;
        case 2:
            measure_time<float>(
                call_v10_base_float, 
                call_v10_sse_float, 
                call_v10_avx_float
            );

            break;
        case 3:
            measure_time<double>(
                call_v10_base_double, 
                call_v10_sse_double, 
                call_v10_avx_double
            );

            break;
        };

        break;
    }
    }
}

void print_mult() {
    wprintf(
        L"Виконуємо операцію %hs для типу %hs з розміром матриці N=%d...\n",
        (unsigned char*)TEXT_variants[op_variant - 1], 
        (unsigned char*)TEXT_types[op_type - 1], 
        op_n
    );

    process_variant();

    wprintf(L"1. Назад\n");
}

const wchar_t* format_status(bool status) {
    return status ? 
        L"\033[32mВвімкнено\033[0m" : 
        L"\033[31mВимкнено\033[0m";
}

void print_settings() {
    wprintf(
        L"Налаштування\n"
        L"1. Запис результату у файл     [%ls]\n"
        L"2. Запис вхідних даних у файл  [%ls]\n"
        L"3. Запис вихідних даних у файл [%ls]\n"
        L"4. Назад\n",
        format_status(SETTING_save_result),
        format_status(SETTING_save_input),
        format_status(SETTING_save_output)
    );
}


int process_input(int action, int& menu_panel) {
    switch (menu_panel)
    {
    case PANEL_main_menu: {
        switch (action)
        {
        case 1:
            menu_panel = PANEL_cpuid;

            break;

        case 2:
            menu_panel = PANEL_input_variant;

            break;

        case 3:
            menu_panel = PANEL_settings;

            break;

        case 4:
            return 1;
        
        default:
            throw L"Такого варіанта не існує!";
        };
        
        break;
    }

    case PANEL_cpuid: {
        switch (action)
        {
        case 1:
            menu_panel = PANEL_main_menu;

            break;

        default:
            throw L"Такого варіанта не існує!";
        };
        
        break;
    }

    case PANEL_input_variant: {
        if (action > 0 && action < 12) {
            if (action == 11)
                menu_panel = PANEL_main_menu;
            else {
                op_variant = action;
                menu_panel = PANEL_input_type;
            }
        }
        else
            throw L"Такого варіанта не існує!";
        
        break;
    }

    case PANEL_input_type: {
        if (action > 0 && action < 5) {
            if (action == 4)
                menu_panel = PANEL_input_variant;
            else {
                op_type = action;
                menu_panel = PANEL_input_n;
            }
        }
        else
            throw L"Такого варіанта не існує!";
        
        break;
    }

    case PANEL_input_n: {
        if (action == 1) {
            menu_panel = PANEL_input_type;

            break;
        }
        
        if (action % 8 != 0)
            throw L"N повинен бути кратним 8!";
        
        op_n = action;
        menu_panel = PANEL_mult;

        break;
    }

    case PANEL_mult: {
        switch (action)
        {
        case 1:
            menu_panel = PANEL_input_n;

            break;

        default:
            throw L"Такого варіанта не існує!";
        };

        break;
    }

    case PANEL_settings: {
        switch (action)
        {
        case 1:
            SETTING_save_result = !SETTING_save_result;

            break;
        case 2:
            SETTING_save_input = !SETTING_save_input;

            break;
        case 3:
            SETTING_save_output = !SETTING_save_output;

            break;
        case 4:
            menu_panel = PANEL_main_menu;

            break;
        
        default:
            throw L"Такого варіанта не існує!";
        };
        
        break;
    }
    };

    return 0;
}


void main_loop() {
    const wchar_t* last_error_text = NULL;
    int menu_panel = PANEL_main_menu;
    
    while (1) {
        try {
            system("cls");

            switch (menu_panel)
            {
            case PANEL_main_menu:
                print_main_menu();

                break;

            case PANEL_cpuid:
                print_cpuid();
                
                break;

            case PANEL_input_variant:
                print_input_variant();
                
                break;
            
            case PANEL_input_type:
                print_input_type();
                
                break;

            case PANEL_input_n:
                print_input_n();
                
                break;

            case PANEL_mult:
                print_mult();
                
                break;

            case PANEL_settings:
                print_settings();
                
                break;
            };

            if (last_error_text) {
                wprintf(L"\033[31mЗауваження: %ls\033[0m\n", last_error_text);
                
                last_error_text = NULL;
            }

            int action = winputd(L"> ");

            if (process_input(action, menu_panel)) return;
        }
        catch (const wchar_t* error_text) {
            last_error_text = error_text;
        }
    }
}


int main(int argc, char* argv[]) {
    srand(time(NULL));
    system("chcp 65001");
    setlocale(LC_ALL, "UTF-8");
    _setmode(_fileno(stdin), _O_U16TEXT);
    _setmode(_fileno(stdout), _O_U16TEXT);

    for (size_t i = 0; i < argc; i++) {
        char* arg = argv[i];

        if (strcmp(arg, "--target") == 0) {
            SETTING_target = true;

            op_variant = atoi(argv[i + 1]);
            op_type = atoi(argv[i + 2]);
            op_n = atoi(argv[i + 3]);
            
            i += 3;
        }
        else if (strcmp(arg, "--save-result") == 0)
            SETTING_save_result = true;
        else if (strcmp(arg, "--save-input") == 0)
            SETTING_save_input = true;
        else if (strcmp(arg, "--save-output") == 0)
            SETTING_save_output = true;
    }

    if (SETTING_target)
        process_variant();
    else
        main_loop();
}
