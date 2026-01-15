with (import <nixpkgs> { config.allowUnfree = true; });

let
  # Combine standard packages with the specific CUDA list from your working file
  LLP = with pkgs; [
    # -- Rust / Build Tools --
    pkg-config
    openssl
    gcc
    cmake
    rustc
    cargo
    rustup
    clang
    libclang
    rust-bindgen

    # -- Audio Dependencies (Required for Jerry) --
    alsa-lib
    alsa-utils
    ffmpeg-full

    # -- Python/AI --
    python312
    python312Packages.torchWithCuda

    # -- CUDA Stack (Matches your udo-env pattern) --
    linuxPackages.nvidia_x11
    cudatoolkit                # The Monolith (Crucial for CMake detection)
    cudaPackages.cuda_cudart
    cudaPackages.cuda_nvcc
    cudaPackages.libcublas
    cudaPackages.cuda_cccl
  ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath LLP;

in
stdenv.mkDerivation {
  name = "jerry-env";
  buildInputs = LLP;
  src = null;

  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)

    # 1. Set Library Path (Matches your working file + Nvidia drivers)
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/run/opengl-driver/lib

    # 2. Set CUDA Roots to the monolithic toolkit (Matches your working file)
    # This is likely the magic bullet. cudatoolkit has the standard layout.
    export CUDA_ROOT=${pkgs.cudatoolkit}
    export CUDAToolkit_ROOT=${pkgs.cudatoolkit}
    export CUDA_PATH=${pkgs.cudatoolkit}

    # 3. Compiler Flags
    # We explicitly point to the monolith's include directory
    export CPATH="${pkgs.cudatoolkit}/include:$CPATH"

    # 4. Bindgen/Clang
    export LIBCLANG_PATH="${pkgs.libclang.lib}/lib"
    export BINDGEN_EXTRA_CLANG_ARGS="-I${pkgs.cudatoolkit}/include"

    export OMP_NUM_THREADS=1
    export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
    mkdir -p $NANOCHAT_BASE_DIR

    echo "Environment loaded using udo-env pattern (Monolithic cudatoolkit)."
  '';
}
