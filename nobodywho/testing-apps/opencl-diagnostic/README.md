# OpenCL loader diagnostic

From a pushed branch, use the existing Mobile Device Tests workflow:

```sh
gh workflow run mobile-device-tests.yml --ref YOUR_BRANCH -f jobs=opencl-diagnostic -f device=both
```

Once the new workflow is on the default branch, you can also use
**Actions → OpenCL Loader Diagnostic → Run workflow**.

The direct and embedded variants run in separate Firebase Test Lab invocations.
Neither loads NobodyWho or downloads a model. The embedded variant links the
same static forwarding shim as production; the direct variant resolves
functions from the device's public `libOpenCL.so` with `dlopen`/`dlsym`.
Both run inside an APK with the same optional native-library declaration.
`device=both` checks Adreno and Mali. The `embedded` mode name is retained
so results can be compared with earlier ICD-loader runs.

In each FTL result's logcat, search for `OpenCLProbe`. Logs include device,
driver, alignment, handles, region sizes and error codes. `-99999` means the
driver left the error sentinel untouched. Missing OpenCL is a failed diagnostic,
not a skipped/passing test. APKs and unstripped native libraries are also uploaded
as GitHub artifacts; Firebase result links are printed in the job logs.

The probe checks 1 MiB sub-buffers at zero and device-aligned offsets, including
a write through the child and read through the parent. Direct-only success
implicates the embedded loading path; failure in both needs driver/argument
investigation. Success in both does **not** clear model loading: reproducing
the actual tensor sizes, offsets and allocation pressure is a follow-up test.

This workflow is manual only; it does not run in nightly or release testing.
