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
cargo build --release --features="serde_support"
cd ../
mkdir -p models
cd models
# Moonshine base (ONNX) - speech-to-text. This is the layout transcribe-rs
# expects (encoder/decoder .onnx + tokenizer.json in models/moonshine-base).
curl -L https://blob.handy.computer/moonshine-base.tar.gz | tar xz
# Smart Turn v3.2 (CPU) - the voice-activity/turn-completion gate.
curl -LO https://raw.githubusercontent.com/pipecat-ai/pipecat/main/src/pipecat/audio/turn/smart_turn/data/smart-turn-v3.2-cpu.onnx
cd ../
cargo run --release --features="serde_support"
