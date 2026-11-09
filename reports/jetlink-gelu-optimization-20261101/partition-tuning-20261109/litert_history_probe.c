// Standalone GPU replay through Jetlink's unmodified LiteRT 2.2 C shim.
#include "jl_litert.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#define MAX_IO 32
#define MAX_FRAMES 160
#define MAX_OWNED (MAX_IO * 3)

typedef struct {
  char name[256];
  int type;
  int64_t dims[16];
  size_t rank, count, bytes;
  int state, pair, write_pending;
  void *host;
  jl_litert_buffer *buffers[2];
} Io;

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec * 1000.0 + (double)t.tv_nsec / 1e6;
}

static int compare_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

static size_t element_bytes(int type) {
  switch (type) {
    case 1: return 4;
    case 10: return 2;
    case 2: case 3: case 9: return 1;
    case 4: case 5: return 2;
    case 6: return 4;
    case 7: case 11: return 8;
    default: return 0;
  }
}

static int describe(jl_litert_model *model, int output, size_t index, Io *io) {
  char *error = jl_litert_model_io_info(model, output, index, io->name, sizeof(io->name),
                                       &io->type, io->dims, 16, &io->rank);
  if (error) {
    fprintf(stderr, "describe: %s\n", error);
    jl_litert_free(error);
    return -1;
  }
  if (io->rank > 16 || !element_bytes(io->type)) return -1;
  io->count = 1;
  for (size_t k = 0; k < io->rank; k++) {
    if (io->dims[k] <= 0 || (uint64_t)io->dims[k] > SIZE_MAX / io->count) return -1;
    io->count *= (size_t)io->dims[k];
  }
  if (io->count > SIZE_MAX / element_bytes(io->type)) return -1;
  io->bytes = io->count * element_bytes(io->type);
  io->pair = -1;
  io->state = !output && !strncmp(io->name, "state_", 6);
  io->host = malloc(io->bytes);
  if (!io->host) return -1;
  fprintf(stderr, "%s %s type=%d bytes=%zu shape=[", output ? "out" : "in", io->name, io->type, io->bytes);
  for (size_t k = 0; k < io->rank; k++) fprintf(stderr, "%s%lld", k ? "," : "", (long long)io->dims[k]);
  fprintf(stderr, "]\n");
  return 0;
}

// -1 invalid, 0 missing state file, 1 loaded. Conversion is only permitted for
// an exact 2x/half byte-size match between known FLOAT/FLOAT16 element types.
static int load_input(Io *io, const char *directory, int frame) {
  char path[1024];
  int used = snprintf(path, sizeof(path), "%s/%d_%s.bin", directory, frame, io->name);
  if (used < 0 || (size_t)used >= sizeof(path)) return -1;
  FILE *file = fopen(path, "rb");
  if (!file) {
    if (io->state && errno == ENOENT) return 0;
    fprintf(stderr, "input open failed: %s: %s\n", path, strerror(errno));
    return -1;
  }
  if (fseek(file, 0, SEEK_END)) { fclose(file); return -1; }
  long length = ftell(file);
  if (length < 0 || fseek(file, 0, SEEK_SET)) { fclose(file); return -1; }
  size_t bytes = (size_t)length;
  int conversion = 0;
  if (bytes != io->bytes) {
    if (io->type == 1 && bytes == io->count * 2) conversion = 1;
    else if (io->type == 10 && bytes == io->count * 4) conversion = 2;
    else if (io->type == 2 && bytes == io->count * 2 &&
             (!strcmp(io->name, "new_img") || !strcmp(io->name, "state_img_q"))) conversion = 3;
    else {
      fprintf(stderr, "input size mismatch: %s expected=%zu got=%zu\n", path, io->bytes, bytes);
      fclose(file);
      return -1;
    }
  }
  void *raw = conversion ? malloc(bytes) : io->host;
  if (!raw) { fclose(file); return -1; }
  size_t got = fread(raw, 1, bytes, file);
  int extra = fgetc(file);
  int read_error = ferror(file);
  fclose(file);
  if (got != bytes || extra != EOF || read_error) {
    if (conversion) free(raw);
    fprintf(stderr, "input read mismatch: %s\n", path);
    return -1;
  }
  if (conversion == 1) {
    const _Float16 *input = raw;
    float *output = io->host;
    for (size_t k = 0; k < io->count; k++) output[k] = (float)input[k];
  } else if (conversion == 2) {
    const float *input = raw;
    _Float16 *output = io->host;
    for (size_t k = 0; k < io->count; k++) output[k] = (_Float16)input[k];
  } else if (conversion == 3) {
    // The existing history export stores original UINT8 image values in FP16.
    // Accept this only for known image inputs, and require an exact round trip.
    const _Float16 *input = raw;
    uint8_t *output = io->host;
    for (size_t k = 0; k < io->count; k++) {
      float v = (float)input[k];
      if (!isfinite(v) || v < 0 || v > 255 || v != floorf(v)) {
        fprintf(stderr, "non-integral/out-of-range UINT8 image source: %s index=%zu value=%g\n", path, k, (double)v);
        free(raw);return -1;
      }
      output[k] = (uint8_t)v;
    }
  }
  if (conversion) {
    if (frame == 0) fprintf(stderr, "source conversion %s: %s\n", io->name,
                            conversion == 1 ? "FLOAT16->FLOAT" : conversion == 2 ? "FLOAT->FLOAT16" : "FLOAT16->UINT8 exact image");
    free(raw);
  }
  return 1;
}

#define CALL(expression) do { \
  char *error_ = (expression); \
  if (error_) { fprintf(stderr, "%s: %s\n", #expression, error_); \
    jl_litert_free(error_); result = 1; goto cleanup; } \
} while (0)

int main(int argc, char **argv) {
  if (argc < 6 || argc > 7) {
    fprintf(stderr, "usage: %s model.tflite nativeLibDir inputDir outputDir frameCount [cacheDir]\n", argv[0]);
    return 2;
  }
  char *end = NULL;
  errno = 0;
  long frames_long = strtol(argv[5], &end, 10);
  if (errno || end == argv[5] || *end || frames_long < 1 || frames_long > MAX_FRAMES) return 2;
  int frames = (int)frames_long, result = 0, phase = 0, main_output = -1;
  size_t ni = 0, no = 0, n_owned = 0;
  jl_litert_model *model = NULL;
  jl_litert_buffer *owned[MAX_OWNED] = {0};
  jl_litert_buffer *inputs[MAX_IO] = {0}, *outputs[MAX_IO] = {0};
  Io in[MAX_IO] = {0}, out[MAX_IO] = {0};
  float *float_output = NULL;
  FILE *csv = NULL;
  double times[MAX_FRAMES], total = 0.0, write_total = 0.0, run_total = 0.0, read_total = 0.0;
  if (mkdir(argv[4], 0755) && errno != EEXIST) { perror("output directory"); return 2; }
  if (argc == 7 && mkdir(argv[6], 0755) && errno != EEXIST) { perror("cache directory"); return 2; }
  CALL(jl_litert_open(argv[2]));
  jl_litert_options options = {.gpu = 1, .cpu_threads = 4,
    .cache_dir = argc == 7 ? argv[6] : NULL,
    .cache_key = argc == 7 ? (getenv("JL_LITERT_CACHE_KEY") ? getenv("JL_LITERT_CACHE_KEY") : "native_gpu_history_20261109") : NULL,
    .npu_cache_dir = NULL};
  CALL(jl_litert_model_create(argv[1], &options, &model));
  int fully = 0;
  CALL(jl_litert_model_fully_accelerated(model, &fully));
  fprintf(stderr, "backend=LiteRT2.2 GPU OpenCL FP16 fully_accelerated=%d\n", fully);
  if (!fully) { fprintf(stderr, "refusing partial acceleration\n"); result = 3; goto cleanup; }
  ni = jl_litert_model_io_count(model, 0);
  no = jl_litert_model_io_count(model, 1);
  if (!ni || !no || ni > MAX_IO || no > MAX_IO) { result = 2; goto cleanup; }
  for (size_t i = 0; i < ni; i++) if (describe(model, 0, i, &in[i])) { result = 2; goto cleanup; }
  for (size_t i = 0; i < no; i++) {
    if (describe(model, 1, i, &out[i])) { result = 2; goto cleanup; }
    if (!strcmp(out[i].name, "outputs")) main_output = (int)i;
  }
  if (main_output < 0 || (out[main_output].type != 1 && out[main_output].type != 10)) {
    fprintf(stderr, "missing float main tensor named outputs\n"); result = 2; goto cleanup;
  }
  for (size_t i = 0; i < ni; i++) {
    if (in[i].state) {
      for (size_t j = 0; j < no; j++) {
        if (!strncmp(out[j].name, "next_", 5) && !strcmp(out[j].name + 5, in[i].name)) {
          if (in[i].type != out[j].type || in[i].bytes != out[j].bytes || in[i].rank != out[j].rank ||
              memcmp(in[i].dims, out[j].dims, in[i].rank * sizeof(int64_t))) {
            fprintf(stderr, "state pair metadata mismatch: %s\n", in[i].name); result = 2; goto cleanup;
          }
          in[i].pair = (int)j;
          out[j].pair = (int)i;
          break;
        }
      }
      if (in[i].pair < 0) { fprintf(stderr, "missing state output: %s\n", in[i].name); result = 2; goto cleanup; }
      int hardware = 0;
      CALL(jl_litert_model_input_hardware(model, i, &hardware));
      fprintf(stderr, "state_managed_pingpong %s preferred_hardware=%d\n", in[i].name, hardware);
    }
    CALL(jl_litert_buffer_create(model, 0, i, &in[i].buffers[0]));
    owned[n_owned++] = in[i].buffers[0];
    if (in[i].state) {
      CALL(jl_litert_buffer_create(model, 0, i, &in[i].buffers[1]));
      owned[n_owned++] = in[i].buffers[1];
    } else in[i].buffers[1] = in[i].buffers[0];
  }
  for (size_t j = 0; j < no; j++) {
    if (out[j].pair >= 0) {
      out[j].buffers[0] = in[out[j].pair].buffers[1];
      out[j].buffers[1] = in[out[j].pair].buffers[0];
    } else {
      CALL(jl_litert_buffer_create(model, 1, j, &out[j].buffers[0]));
      owned[n_owned++] = out[j].buffers[0];
      out[j].buffers[1] = out[j].buffers[0];
    }
  }
  Io *main = &out[main_output];
  float_output = malloc(main->count * sizeof(float));
  if (!float_output) { result = 2; goto cleanup; }
  char csv_path[1024];
  int used = snprintf(csv_path, sizeof(csv_path), "%s/frames.csv", argv[4]);
  if (used < 0 || (size_t)used >= sizeof(csv_path)) { result = 2; goto cleanup; }
  csv = fopen(csv_path, "w");
  if (!csv) { result = 2; goto cleanup; }
  fprintf(csv, "frame,total_ms,input_write_ms,run_ms,main_read_ms,nonfinite\n");
  fprintf(stderr, "timing=input-buffer writes+run+main-output read; file reads, source conversions and dumps excluded\n");
  for (int frame = 0; frame < frames; frame++) {
    for (size_t i = 0; i < ni; i++) {
      int loaded = load_input(&in[i], argv[3], frame);
      if (loaded < 0) { result = 4; goto cleanup; }
      in[i].write_pending = loaded;
      inputs[i] = in[i].buffers[phase];
    }
    for (size_t j = 0; j < no; j++) outputs[j] = out[j].buffers[phase];
    double started = now_ms();
    for (size_t i = 0; i < ni; i++) if (in[i].write_pending)
      CALL(jl_litert_buffer_write(inputs[i], in[i].host, in[i].bytes));
    double written = now_ms();
    CALL(jl_litert_run(model, inputs, ni, outputs, no));
    double ran = now_ms();
    CALL(jl_litert_buffer_read(outputs[main_output], main->host, main->bytes));
    double read = now_ms();
    phase ^= 1;
    times[frame] = read - started;
    total += times[frame]; write_total += written - started; run_total += ran - written; read_total += read - ran;
    size_t nonfinite = 0;
    for (size_t k = 0; k < main->count; k++) {
      float value = main->type == 1 ? ((const float *)main->host)[k] : (float)((const _Float16 *)main->host)[k];
      float_output[k] = value;
      if (!isfinite(value)) nonfinite++;
    }
    char path[1024];
    used = snprintf(path, sizeof(path), "%s/out_%d.bin", argv[4], frame);
    if (used < 0 || (size_t)used >= sizeof(path)) { result = 2; goto cleanup; }
    FILE *file = fopen(path, "wb");
    if (!file) { result = 5; goto cleanup; }
    size_t wrote = fwrite(float_output, sizeof(float), main->count, file);
    int close_error = fclose(file);
    if (wrote != main->count || close_error) { result = 5; goto cleanup; }
    fprintf(csv, "%d,%.6f,%.6f,%.6f,%.6f,%zu\n", frame, times[frame], written - started, ran - written, read - ran, nonfinite);
    fflush(csv);
    fprintf(stderr, "frame=%d total_ms=%.3f write=%.3f run=%.3f read=%.3f nonfinite=%zu/%zu\n",
            frame, times[frame], written - started, ran - written, read - ran, nonfinite, main->count);
    if (nonfinite) { result = 6; goto cleanup; }
  }
  qsort(times, (size_t)frames, sizeof(double), compare_double);
  printf("backend=LiteRT_GPU_OpenCL fully_accelerated=1 frames=%d mean_ms=%.3f p50_ms=%.3f p99_ms=%.3f min_ms=%.3f max_ms=%.3f mean_write_ms=%.3f mean_run_ms=%.3f mean_read_ms=%.3f output_float32_count=%zu\n",
         frames, total / frames, times[(frames - 1) / 2], times[(frames - 1) * 99 / 100], times[0], times[frames - 1],
         write_total / frames, run_total / frames, read_total / frames, main->count);
cleanup:
  if (csv) fclose(csv);
  for (size_t i = 0; i < n_owned; i++) jl_litert_buffer_release(owned[i]);
  for (size_t i = 0; i < MAX_IO; i++) { free(in[i].host); free(out[i].host); }
  free(float_output);
  jl_litert_model_release(model);
  return result;
}
