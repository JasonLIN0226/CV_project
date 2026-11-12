# Sourced by burst/run_here.sh. Makes the GPU usable inside an OOD session container.
# The container exposes /dev/nvidia* but not the NVIDIA user-space libraries, so PyTorch reports
# no GPU. This unpacks the driver package that matches the kernel module into
# ~/nvidia-driver-<version> (once) and puts it on LD_LIBRARY_PATH. Batch jobs do not need it.
if python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "GPU visible to PyTorch"
elif [ ! -r "${BM_NVIDIA_VERSION_FILE:=/proc/driver/nvidia/version}" ]; then
    echo "no NVIDIA device in this session: start a session on the GPU partition, or submit with burst/submit.sh" >&2
    return 1 2>/dev/null || exit 1
else
    # first dotted number of the NVRM line, e.g. "... Kernel Module  610.43.02  Thu ..." -> 610.43.02
    _bm_version=$(grep NVRM "$BM_NVIDIA_VERSION_FILE" | grep -oE "[0-9]+\.[0-9]+(\.[0-9]+)?" | head -1)
    [ -n "$_bm_version" ] || { echo "cannot read the driver version from $BM_NVIDIA_VERSION_FILE" >&2; return 1 2>/dev/null || exit 1; }
    _bm_driver="$HOME/nvidia-driver-$_bm_version"
    if [ ! -d "$_bm_driver" ]; then
        echo "fetching NVIDIA libraries $_bm_version into $_bm_driver"
        _bm_tmp=$(mktemp -d)
        curl -fsSL -o "$_bm_tmp/driver.run" "https://download.nvidia.com/XFree86/Linux-x86_64/$_bm_version/NVIDIA-Linux-x86_64-$_bm_version.run"
        sh "$_bm_tmp/driver.run" --extract-only --target "$_bm_driver" >/dev/null
        for _bm_lib in "$_bm_driver"/lib*.so."$_bm_version"; do
            ln -sf "$(basename "$_bm_lib")" "${_bm_lib%."$_bm_version"}.1"
        done
        rm -rf "$_bm_tmp"
    fi
    export LD_LIBRARY_PATH="$_bm_driver${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    python -c "import torch, sys; print('GPU visible to PyTorch:', torch.cuda.get_device_name(0)) if torch.cuda.is_available() else sys.exit('PyTorch still sees no GPU; use burst/submit.sh instead')"
    unset _bm_version _bm_driver _bm_tmp _bm_lib
fi
