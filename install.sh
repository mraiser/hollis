DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR
git clone https://github.com/mraiser/newbound.git CHUCKTHIS
mkdir -p src
cp -r CHUCKTHIS/src/* src/
cp -r CHUCKTHIS/data/* data/
mkdir -p newbound_core
cp -r CHUCKTHIS/newbound_core/* newbound_core/
cp CHUCKTHIS/Cargo.toml Cargo.toml
rm -f CHUCKTHIS/Cargo.lock
rm -f Cargo.lock
rm -rf cmd
rustup update
cd CHUCKTHIS
cargo build --release --features="serde_support"
cd ../
CHUCKTHIS/target/release/newbound rebuild
rm -rf CHUCKTHIS
cd hollis
cargo build --release
cd ../
mkdir -p models
cd models
wget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin
cd ../
cargo run --release --features="serde_support"
