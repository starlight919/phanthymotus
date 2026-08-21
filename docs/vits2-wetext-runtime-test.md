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

The pinned ModelScope revision includes a checksum-verified `tn_manifest.json`
and both TN FSTs; its expected `20dB` output is `二十分贝`. The VITS2 release is
downloaded lazily by the service, so a validation must call the production
downloader before it inspects `/models/vits2/tn_cache`. An empty cache before
that call is not a missing-release error and does not require republishing the
model.

The adapter retains legacy metadata handling only for backwards compatibility;
the current validation requires the manifest contract and does not accept an
unmanifested release.

## JP5 Python compatibility

JP5.1.1 uses Python 3.8. One of the legacy frontend dependencies still uses
the removed `np.bool` alias, so the shared requirements file pins
`numpy==1.23.5` for Python < 3.9 and retains `numpy==1.24.4` for JP6/Python
3.9+. This is an ABI/runtime compatibility constraint, not a model or TensorRT
difference. The Docker build applies the matching marker automatically.

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
