//! Build script for Python bindings
//!
//! This script configures PyO3 to properly link against Python libraries

fn main() {
    // Use pyo3-build-config to automatically detect Python
    pyo3_build_config::use_pyo3_cfgs();
    
    // On macOS with pyenv, we may need additional linking configuration
    #[cfg(target_os = "macos")]
    {
        // Try to find Python library path
        if let Ok(python) = std::env::var("PYO3_PYTHON") {
            // Get library directory (where libpython3.11.dylib is)
            if let Ok(output) = std::process::Command::new(&python)
                .arg("-c")
                .arg("import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
                .output()
            {
                if let Ok(libdir) = String::from_utf8(output.stdout) {
                    let libdir = libdir.trim();
                    if !libdir.is_empty() {
                        println!("cargo:rustc-link-search=native={}", libdir);
                    }
                }
            }
            
            // Get config directory (where libpython3.11.a might be)
            if let Ok(output) = std::process::Command::new(&python)
                .arg("-c")
                .arg("import sysconfig; print(sysconfig.get_config_var('LIBPL'))")
                .output()
            {
                if let Ok(libpl) = String::from_utf8(output.stdout) {
                    let libpl = libpl.trim();
                    if !libpl.is_empty() {
                        println!("cargo:rustc-link-search=native={}", libpl);
                    }
                }
            }
            
            // Try to link against python3.11 (without lib prefix and extension)
            // The linker will automatically find libpython3.11.dylib or libpython3.11.a
            if let Ok(output) = std::process::Command::new(&python)
                .arg("--version")
                .output()
            {
                if let Ok(version) = String::from_utf8(output.stdout) {
                    // Extract version like "3.11.13" -> "3.11"
                    if let Some(version_part) = version.split_whitespace().nth(1) {
                        if let Some(major_minor) = version_part.split('.').take(2).collect::<Vec<_>>().join(".").as_str().get(0..) {
                            let libname = format!("python{}", major_minor);
                            println!("cargo:rustc-link-lib=dylib={}", libname);
                        }
                    }
                }
            }
        }
    }
}

