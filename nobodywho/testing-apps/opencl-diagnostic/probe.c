// Same operations, independently built with direct or embedded OpenCL loading.
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <android/log.h>
#include <dlfcn.h>
#include <jni.h>
#include <stdlib.h>
#include <string.h>

#define LOG(...) __android_log_print(ANDROID_LOG_INFO, "OpenCLProbe", __VA_ARGS__)
#define SENTINEL (-99999)
#ifdef DIRECT_OPENCL
#define MODE "direct"
#define LOAD(fn) __typeof__(&fn) p_##fn = (__typeof__(&fn))dlsym(library, #fn); \
    if (!p_##fn) { LOG("missing %s: %s", #fn, dlerror()); dlclose(library); return JNI_FALSE; }
#else
#define MODE "embedded"
#define LOAD(fn) __typeof__(&fn) p_##fn = &fn
#endif
#define CHECK(call) do { cl_int check_status = (call); if (check_status != CL_SUCCESS) { \
    LOG("%s returned %d", #call, check_status); goto cleanup; } } while (0)

JNIEXPORT jboolean JNICALL Java_ai_nobodywho_opencl_Probe_run(JNIEnv *env, jclass cls) {
    (void)env; (void)cls;
    LOG("BEGIN mode=%s", MODE);
    setenv("OCL_ICD_ENABLE_TRACE", "1", 1);
    void *library = NULL;
#ifdef DIRECT_OPENCL
    library = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!library) { LOG("dlopen failed: %s", dlerror()); return JNI_FALSE; }
#endif
    LOAD(clGetPlatformIDs); LOAD(clGetDeviceIDs); LOAD(clGetDeviceInfo);
    LOAD(clCreateContext); LOAD(clCreateCommandQueue); LOAD(clCreateBuffer);
    LOAD(clCreateSubBuffer); LOAD(clGetMemObjectInfo);
    LOAD(clEnqueueWriteBuffer); LOAD(clEnqueueReadBuffer);
    LOAD(clReleaseMemObject); LOAD(clReleaseCommandQueue); LOAD(clReleaseContext);

    jboolean passed = JNI_FALSE;
    cl_platform_id *platforms = NULL;
    cl_context context = NULL;
    cl_command_queue queue = NULL;
    cl_mem parent = NULL, child = NULL;
    cl_device_id device = NULL;
    cl_uint count = 0;
    CHECK(p_clGetPlatformIDs(0, NULL, &count));
    if (!count) { LOG("no OpenCL platforms"); goto cleanup; }
    platforms = calloc(count, sizeof(*platforms));
    if (!platforms) goto cleanup;
    CHECK(p_clGetPlatformIDs(count, platforms, NULL));
    for (cl_uint i = 0; i < count; ++i) {
        cl_int status = p_clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        if (status == CL_DEVICE_NOT_FOUND) continue;
        CHECK(status);
        break;
    }
    if (!device) { LOG("no OpenCL GPU"); goto cleanup; }
    char name[256] = {0}, driver[256] = {0};
    cl_uint alignment_bits = 0;
    CHECK(p_clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(name), name, NULL));
    CHECK(p_clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(driver), driver, NULL));
    CHECK(p_clGetDeviceInfo(device, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(alignment_bits), &alignment_bits, NULL));
    LOG("mode=%s device=%s driver=%s alignment_bits=%u", MODE, name, driver, alignment_bits);
    cl_int err = SENTINEL;
    context = p_clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    LOG("context=%p err=%d", (void *)context, err);
    if (!context || err != CL_SUCCESS) goto cleanup;
    err = SENTINEL;
    queue = p_clCreateCommandQueue(context, device, 0, &err);
    LOG("queue=%p err=%d", (void *)queue, err);
    if (!queue || err != CL_SUCCESS) goto cleanup;

    size_t alignment = (alignment_bits + 7u) / 8u;
    size_t size = 1024 * 1024, parent_size = size + alignment;
    err = SENTINEL;
    parent = p_clCreateBuffer(context, CL_MEM_READ_WRITE, parent_size, NULL, &err);
    LOG("parent=%p size=%zu err=%d", (void *)parent, parent_size, err);
    if (!parent || err != CL_SUCCESS) goto cleanup;
    for (int i = 0; i < 2; ++i) {
        cl_buffer_region region = {i == 0 ? 0 : alignment, size};
        err = SENTINEL;
        child = p_clCreateSubBuffer(parent, CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION, &region, &err);
        LOG("subbuffer=%p origin=%zu size=%zu err=%d sentinel=%d", (void *)child, region.origin, region.size, err, SENTINEL);
        if (!child || err != CL_SUCCESS) {
            // An ABI mismatch can return an integer error as a bogus handle.
            // Never pass an unsuccessfully created child to the release API.
            child = NULL;
            goto cleanup;
        }
        size_t actual_size = 0;
        CHECK(p_clGetMemObjectInfo(child, CL_MEM_SIZE, sizeof(actual_size), &actual_size, NULL));
        if (actual_size != size) { LOG("unexpected size=%zu", actual_size); goto cleanup; }
        unsigned char input[] = {1, 2, 3, 4}, output[sizeof(input)] = {0};
        CHECK(p_clEnqueueWriteBuffer(queue, child, CL_TRUE, 0, sizeof(input), input, 0, NULL, NULL));
        CHECK(p_clEnqueueReadBuffer(queue, parent, CL_TRUE, region.origin, sizeof(output), output, 0, NULL, NULL));
        if (memcmp(input, output, sizeof(input))) { LOG("data mismatch"); goto cleanup; }
        CHECK(p_clReleaseMemObject(child));
        child = NULL;
    }
    passed = JNI_TRUE;
cleanup:
    if (child) LOG("release child: %d", p_clReleaseMemObject(child));
    if (parent) LOG("release parent: %d", p_clReleaseMemObject(parent));
    if (queue) LOG("release queue: %d", p_clReleaseCommandQueue(queue));
    if (context) LOG("release context: %d", p_clReleaseContext(context));
    free(platforms);
    if (library) dlclose(library);
    LOG("END mode=%s result=%s", MODE, passed ? "PASS" : "FAIL");
    return passed;
}
