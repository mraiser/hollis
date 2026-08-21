// This file is auto-generated and managed by the flowlang build script.
use flowlang::rustcmd::Transform;
pub mod audio;
pub mod hollis;
pub fn cmdinit(cmds: &mut Vec<(String, Transform, String)>) {
    hollis::cmdinit(cmds);
    audio::cmdinit(cmds);
}
