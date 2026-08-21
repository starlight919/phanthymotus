# VITS2 WeText Runtime Test

This branch tests the VITS2 frontend with the published `wetext` package and
its lightweight `kaldifst` runtime. It intentionally does not compile or
install OpenFST/Pynini in the Jetson Docker build.

JP5.1.1 uses Python 3.8. `wetext==0.1.6` imports the Python 3.9-only
`importlib.resources.files`, despite declaring Python 3.7 support. Only on
Python <3.9, the Docker build installs the standard `importlib_resources`
backport and applies the equivalent import fallback before importing WeText.
JP6 leaves the published package untouched. This is a Python-version
compatibility patch only; it does not alter FST graphs or TN behavior.

`onnxruntime` remains installed because it is used by the existing perception
runtime. TensorRT plans remain external model artifacts and are mounted or
downloaded through the normal VITS2 model-release path.

The plugin loads the model release's checksum-verified legacy `tn_cache` FSTs
directly through `kaldifst`, with WeText's token parser providing the existing
tagger-to-verbalizer ordering. No private wheel or source archive is fetched.
This adapter is intentionally a test implementation; after native validation,
the same interface should move upstream into WeText.

The current V3 80k model revision predates `tn_manifest.json` and supplies
`SHA256SUMS`; the adapter validates that legacy contract as well. Its `20dB`
output is the historical `二十dB`. A corrected `二十分贝` graph must be shipped
as a new ModelScope model revision with its matching manifest, never injected
by the runtime.

Some historical ModelScope manifests omitted both TN metadata files from their
runtime-required list although they include the two FST files. The test adapter
accepts that legacy layout only to prove graph readability and prints
`tn_release_contract=unmanifested`; it is not a checksum-verifiable production
contract. The next model revision must list and download `tn_manifest.json`.

Build a JP6.1 image on a native Jetson with:

```bash
docker build --network host \
  --build-arg JP_VERSION=61 \
  -f perception/Dockerfile.jetson \
  -t phanthymotus-perception:vits2-wetext-jp61 .
```

The workspace validation entrypoint is:

```bash
bash /workspace/VITS-dev/VITS_Jetson/ops/jetson/build_test_vits2_kaldifst_jp6_monitored.sh
```

It performs the normal MCP-to-ROS audio, RTF and container-memory checks and
also asserts that Pynini is absent while the checksum-verified external TN
release handles the focused acronym/unit cases.

Local parity is complete: 102 focused cases and 16,008 historical texts match
the legacy Pynini frontend exactly. The branch remains a frontend/runtime
feasibility test until it passes the normal
MCP-to-ROS audio regression before its normalizer output replaces a production
frontend, because published `wetext` output is not assumed to be token-identical
to the existing customized TN assets.
