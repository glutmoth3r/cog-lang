//! Deterministic source-package archives assembled from the resolved project graph.
use crate::config::{self, Package};
use crate::project::{self, Prepared};
use crate::resolver::Source;
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const MAX_ARCHIVE_ENTRIES: usize = 65_536;
const MAX_ARCHIVE_HEADERS: usize = MAX_ARCHIVE_ENTRIES * 2;
const MAX_ARCHIVE_ENTRY_PATH_BYTES: usize = 4096;
const MAX_ARCHIVE_PATH_BYTES: usize = 64 * 1024 * 1024;
const MAX_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_COMPRESSED_ARCHIVE_BYTES: u64 = 512 * 1024 * 1024;
const ZSTD_COMPRESSION_LEVEL: i32 = 6;
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Find an installed COGPKG 2 library in the user-local package store.
pub(crate) fn installed_library_archive(name: &str, version: &str) -> Result<PathBuf, String> {
    if !config::valid_package_name(name) || version.is_empty() {
        return Err("installed package dependency requires a valid name and version".into());
    }
    let home = user_store_root()?;
    let library_root = home.join("libraries");
    let name_root = library_root.join(store_component(name));
    let version_root = name_root.join(store_component(version));
    for directory in [&home, &library_root, &name_root, &version_root] {
        match std::fs::symlink_metadata(directory) {
            Ok(metadata) if !metadata.file_type().is_symlink() && metadata.is_dir() => {}
            Ok(_) => {
                return Err(format!(
                    "installed package store path {} is not a regular directory",
                    directory.display()
                ));
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                return Err(format!(
                    "package `{name}` version `{version}` is not installed"
                ));
            }
            Err(error) => {
                return Err(format!("cannot inspect installed package store: {error}"));
            }
        }
    }
    let archive = version_root.join("package.cogpkg");
    match std::fs::symlink_metadata(&archive) {
        Ok(metadata)
            if !metadata.file_type().is_symlink()
                && metadata.is_file()
                && metadata.len() <= MAX_COMPRESSED_ARCHIVE_BYTES =>
        {
            Ok(archive)
        }
        Ok(_) => Err(format!(
            "installed package archive {} is not a bounded regular file",
            archive.display()
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Err(format!(
            "package `{name}` version `{version}` is not installed"
        )),
        Err(error) => Err(format!("cannot inspect installed package archive: {error}")),
    }
}

/// Resolve a local `.cogpkg` dependency into an immutable content-addressed
/// project cache. Only regular files and directories are accepted from TAR.
pub(crate) fn extract_dependency_archive(
    project_root: &Path,
    archive_path: &Path,
) -> Result<PathBuf, String> {
    extract_archive_cached(
        project_root,
        archive_path,
        ".cog-package-cache",
        validate_cached_package,
    )
}

fn extract_application_archive(archive_path: &Path) -> Result<PathBuf, String> {
    let project_root = archive_path.parent().unwrap_or(Path::new("."));
    extract_archive_cached(
        project_root,
        archive_path,
        ".cog-application-cache",
        |path| validate_application_cache_directory(project_root, path),
    )
}

fn extract_archive_cached(
    project_root: &Path,
    archive_path: &Path,
    cache_name: &str,
    validate: impl Fn(&Path) -> Result<(), String>,
) -> Result<PathBuf, String> {
    let archive_metadata = std::fs::symlink_metadata(archive_path).map_err(|error| {
        format!(
            "cannot inspect package archive {}: {error}",
            archive_path.display()
        )
    })?;
    if archive_metadata.file_type().is_symlink() || !archive_metadata.is_file() {
        return Err(format!(
            "package archive {} is not a regular file",
            archive_path.display()
        ));
    }
    if archive_metadata.len() > MAX_COMPRESSED_ARCHIVE_BYTES {
        return Err(format!(
            "compressed package archive byte limit ({MAX_COMPRESSED_ARCHIVE_BYTES}) exceeded"
        ));
    }
    let mut file = File::open(archive_path).map_err(|error| {
        format!(
            "cannot open package archive {}: {error}",
            archive_path.display()
        )
    })?;
    let mut hasher = blake3::Hasher::new();
    let mut archive_bytes = 0u64;
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            format!(
                "cannot read package archive {}: {error}",
                archive_path.display()
            )
        })?;
        if count == 0 {
            break;
        }
        archive_bytes = archive_bytes
            .checked_add(count as u64)
            .filter(|bytes| *bytes <= MAX_COMPRESSED_ARCHIVE_BYTES)
            .ok_or_else(|| {
                format!(
                    "compressed package archive byte limit ({MAX_COMPRESSED_ARCHIVE_BYTES}) exceeded"
                )
            })?;
        hasher.update(&buffer[..count]);
    }
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        format!(
            "cannot rewind package archive {}: {error}",
            archive_path.display()
        )
    })?;
    if archive_bytes != archive_metadata.len() {
        return Err(format!(
            "package archive {} changed while being read",
            archive_path.display()
        ));
    }
    let digest = hasher.finalize();
    let digest_bytes = *digest.as_bytes();

    let target = project_root.join("target");
    ensure_cache_directory(&target)?;
    let cache_parent = target.join(cache_name);
    ensure_cache_directory(&cache_parent)?;
    let cache_path = cache_parent.join(digest.to_hex().as_str());
    match std::fs::symlink_metadata(&cache_path) {
        Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_dir() => {
            return Err(format!(
                "package cache path {} is not a directory",
                cache_path.display()
            ));
        }
        Ok(_) => {
            validate(&cache_path)?;
            return Ok(cache_path);
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(format!("cannot inspect package cache: {error}"));
        }
    }

    let temporary = loop {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let candidate = cache_parent.join(format!(".extract-{}-{suffix}", std::process::id()));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!("cannot create package cache directory: {error}"));
            }
        }
    };
    let (snapshot_path, snapshot_file) =
        match snapshot_archive(file, &cache_parent, digest_bytes, archive_bytes) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let _ = std::fs::remove_dir_all(&temporary);
                return Err(error);
            }
        };
    let (tar_snapshot_path, tar_snapshot_file) =
        match decompress_archive(snapshot_file, &cache_parent) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                let cleanup = std::fs::remove_file(&snapshot_path);
                let _ = std::fs::remove_dir_all(&temporary);
                return match cleanup {
                    Ok(()) => Err(error),
                    Err(cleanup) => Err(format!(
                        "{error}; cannot remove compressed package snapshot {}: {cleanup}",
                        snapshot_path.display()
                    )),
                };
            }
        };
    let extracted = extract_archive_into(tar_snapshot_file, &temporary).and_then(|()| {
        validate(&temporary)?;
        install_cache_directory(&temporary, &cache_path, validate)
    });
    let compressed_cleanup = std::fs::remove_file(&snapshot_path);
    let tar_cleanup = std::fs::remove_file(&tar_snapshot_path);
    if let Err(error) = extracted {
        let _ = std::fs::remove_dir_all(&temporary);
        let mut message = error;
        if let Err(cleanup) = compressed_cleanup {
            message.push_str(&format!(
                "; cannot remove compressed package snapshot {}: {cleanup}",
                snapshot_path.display()
            ));
        }
        if let Err(cleanup) = tar_cleanup {
            message.push_str(&format!(
                "; cannot remove TAR snapshot {}: {cleanup}",
                tar_snapshot_path.display()
            ));
        }
        return Err(message);
    }
    let _ = std::fs::remove_dir_all(&temporary);
    compressed_cleanup.map_err(|error| {
        format!(
            "cannot remove compressed package snapshot {}: {error}",
            snapshot_path.display()
        )
    })?;
    tar_cleanup.map_err(|error| {
        format!(
            "cannot remove TAR snapshot {}: {error}",
            tar_snapshot_path.display()
        )
    })?;
    Ok(cache_path)
}

fn decompress_archive(source: File, directory: &Path) -> Result<(PathBuf, File), String> {
    decompress_archive_with_limit(source, directory, MAX_ARCHIVE_BYTES)
}

fn decompress_archive_with_limit(
    source: File,
    directory: &Path,
    limit: u64,
) -> Result<(PathBuf, File), String> {
    let (path, output) = loop {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let candidate = directory.join(format!(".tar-{}-{suffix}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot create TAR snapshot: {error}")),
        }
    };
    let result = (|| {
        let mut decoder = zstd::stream::read::Decoder::new(source)
            .map_err(|error| format!("invalid Zstandard package frame: {error}"))?
            .single_frame();
        let mut output = CountingWriter::new(output, limit);
        io::copy(&mut decoder, &mut output)
            .map_err(|error| format!("cannot decompress package TAR: {error}"))?;
        let tar_bytes = output.bytes;
        output
            .inner
            .sync_all()
            .map_err(|error| format!("cannot sync decompressed TAR snapshot: {error}"))?;
        let mut trailing = decoder.finish();
        let mut byte = [0u8; 1];
        if trailing
            .read(&mut byte)
            .map_err(|error| format!("cannot verify package frame end: {error}"))?
            != 0
        {
            return Err("data follows the single Zstandard package frame".into());
        }
        if tar_bytes == 0 {
            return Err("empty Zstandard package frame".into());
        }
        Ok(())
    })();
    if let Err(error) = result {
        let _ = std::fs::remove_file(&path);
        return Err(error);
    }
    let file = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            let _ = std::fs::remove_file(&path);
            return Err(format!("cannot reopen decompressed TAR snapshot: {error}"));
        }
    };
    Ok((path, file))
}

fn snapshot_archive(
    mut source: File,
    directory: &Path,
    expected_digest: [u8; 32],
    expected_bytes: u64,
) -> Result<(PathBuf, File), String> {
    let (path, mut snapshot) = loop {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let candidate = directory.join(format!(".archive-{}-{suffix}", std::process::id()));
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(format!("cannot snapshot package archive: {error}")),
        }
    };
    let result = (|| {
        let mut hasher = blake3::Hasher::new();
        let mut copied = 0u64;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let count = source
                .read(&mut buffer)
                .map_err(|error| format!("cannot reread package archive: {error}"))?;
            if count == 0 {
                break;
            }
            copied = copied
                .checked_add(count as u64)
        .filter(|bytes| *bytes <= MAX_COMPRESSED_ARCHIVE_BYTES)
        .ok_or_else(|| {
            format!(
                "compressed package archive byte limit ({MAX_COMPRESSED_ARCHIVE_BYTES}) exceeded"
            )
        })?;
            snapshot
                .write_all(&buffer[..count])
                .map_err(|error| format!("cannot snapshot package archive: {error}"))?;
            hasher.update(&buffer[..count]);
        }
        if copied != expected_bytes || hasher.finalize().as_bytes() != &expected_digest {
            return Err("package archive changed while being snapshotted".into());
        }
        snapshot
            .sync_all()
            .map_err(|error| format!("cannot sync package archive snapshot: {error}"))
    })();
    if let Err(error) = result {
        drop(snapshot);
        let _ = std::fs::remove_file(&path);
        return Err(error);
    }
    drop(snapshot);
    let snapshot = match File::open(&path) {
        Ok(file) => file,
        Err(error) => {
            let _ = std::fs::remove_file(&path);
            return Err(format!("cannot reopen package archive snapshot: {error}"));
        }
    };
    Ok((path, snapshot))
}

fn ensure_cache_directory(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_symlink() && metadata.is_dir() => Ok(()),
        Ok(_) => Err(format!(
            "package cache parent {} is not a regular directory",
            path.display()
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => std::fs::create_dir(path)
            .map_err(|error| format!("cannot create package cache directory: {error}")),
        Err(error) => Err(format!("cannot inspect package cache directory: {error}")),
    }
}

fn validate_cached_package(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_symlink() && metadata.is_dir() => {}
        Ok(_) => {
            return Err(format!(
                "package cache path {} is not a regular directory",
                path.display()
            ));
        }
        Err(error) => {
            return Err(format!(
                "cannot inspect package cache path {}: {error}",
                path.display()
            ));
        }
    }
    let marker = path.join("COGPKG");
    let marker_metadata = std::fs::symlink_metadata(&marker)
        .map_err(|error| format!("package archive has no format marker: {error}"))?;
    if marker_metadata.file_type().is_symlink() || !marker_metadata.is_file() {
        return Err("package archive format marker is not a regular file".into());
    }
    const FORMAT_MARKER: &[u8] = b"COGPKG 2\n";
    let marker_contents = crate::limits::read_at_most(
        File::open(&marker)
            .map_err(|error| format!("cannot read package archive format marker: {error}"))?,
        FORMAT_MARKER.len() + 1,
    )
    .map_err(|error| format!("cannot read package archive format marker: {error}"))?;
    if marker_metadata.len() != FORMAT_MARKER.len() as u64 || marker_contents != FORMAT_MARKER {
        return Err("unsupported or invalid .cogpkg format version".into());
    }
    let manifest = path.join("package.kdl");
    match std::fs::symlink_metadata(&manifest) {
        Ok(metadata) if !metadata.file_type().is_symlink() && metadata.is_file() => {
            let package =
                config::package(&manifest).map_err(|errors| crate::frontend::render(&errors))?;
            if package.kind != config::PackageKind::Library {
                return Err("local .cogpkg dependencies must be COGPKG 2 library packages".into());
            }
            Ok(())
        }
        Ok(_) => Err(format!(
            "package archive {} has no regular root package.kdl",
            path.display()
        )),
        Err(error) => Err(format!(
            "package archive {} has no root package.kdl: {error}",
            path.display()
        )),
    }
}

fn validate_application_cache_directory(project_root: &Path, path: &Path) -> Result<(), String> {
    let directory = open_application_cache_directory(project_root, path)?;
    validate_application_package_directory(&directory)
}

fn validate_application_package_directory(directory: &PackageDirectory) -> Result<(), String> {
    const FORMAT_MARKER: &[u8] = b"COGPKG 3\n";
    let contents = crate::limits::read_at_most(
        directory
            .open_regular_file(Path::new("COGPKG"))
            .map_err(|error| format!("cannot read package archive format marker: {error}"))?,
        FORMAT_MARKER.len() + 1,
    )
    .map_err(|error| format!("cannot read package archive format marker: {error}"))?;
    if contents != FORMAT_MARKER {
        return Err("unsupported or invalid application .cogpkg format version".into());
    }
    let package = read_package_manifest(directory)?;
    if package.kind != config::PackageKind::Application {
        return Err("COGPKG 3 must contain an application package".into());
    }
    directory
        .open_regular_file(Path::new("cog.lock"))
        .map_err(|error| format!("invalid application package lockfile: {error}"))?;
    directory
        .open_child_directory(Path::new(".cog"))
        .map_err(|error| format!("invalid application package entry .cog: {error}"))?;
    directory
        .open_regular_file(Path::new(".cog/app.cogb"))
        .map_err(|error| format!("invalid application package entry .cog/app.cogb: {error}"))?;
    Ok(())
}

struct PackageDirectory {
    path: PathBuf,
    directory: cap_std::fs::Dir,
}

impl PackageDirectory {
    fn open_ambient(path: &Path) -> Result<Self, String> {
        let directory = cap_std::fs::Dir::open_ambient_dir(path, cap_std::ambient_authority())
            .map_err(|error| {
                format!("cannot open package directory {}: {error}", path.display())
            })?;
        Ok(Self {
            path: path.to_path_buf(),
            directory,
        })
    }

    fn try_clone(&self) -> Result<Self, String> {
        Ok(Self {
            path: self.path.clone(),
            directory: self
                .directory
                .try_clone()
                .map_err(|error| format!("cannot retain package directory handle: {error}"))?,
        })
    }

    fn open_child_directory(&self, relative: &Path) -> Result<Self, String> {
        let relative = normalized_relative(relative)?;
        let mut directory = self
            .directory
            .try_clone()
            .map_err(|error| format!("cannot retain package directory handle: {error}"))?;
        let mut path = self.path.clone();
        for component in relative.components() {
            let Component::Normal(name) = component else {
                return Err("application package dependency path is not normalized".into());
            };
            path.push(name);
            let metadata = directory.symlink_metadata(name).map_err(|error| {
                format!(
                    "application package dependency directory {} is missing: {error}",
                    path.display()
                )
            })?;
            if metadata.file_type().is_symlink() {
                return Err(format!(
                    "application package dependency path {} is a symbolic link",
                    path.display()
                ));
            }
            if !metadata.is_dir() {
                return Err(format!(
                    "application package dependency path {} is not a directory",
                    path.display()
                ));
            }
            directory = directory.open_dir(name).map_err(|error| {
                format!(
                    "cannot open application package dependency directory {}: {error}",
                    path.display()
                )
            })?;
        }
        Ok(Self { path, directory })
    }

    fn open_regular_file(&self, relative: &Path) -> Result<File, String> {
        let relative = normalized_relative(relative)?;
        let mut components = relative.components().peekable();
        let mut parent = self.try_clone()?;
        let mut file_name = None;
        while let Some(component) = components.next() {
            let Component::Normal(name) = component else {
                return Err("package input path must be normalized".into());
            };
            if components.peek().is_some() {
                parent = parent.open_child_directory(Path::new(name))?;
            } else {
                file_name = Some(name.to_os_string());
            }
        }
        let file_name =
            file_name.ok_or_else(|| "package input path must not be empty".to_owned())?;
        let metadata = parent
            .directory
            .symlink_metadata(&file_name)
            .map_err(|error| {
                format!(
                    "cannot inspect package input {}: {error}",
                    parent.path.join(&file_name).display()
                )
            })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "package input {} is a symbolic link",
                parent.path.join(&file_name).display()
            ));
        }
        if !metadata.is_file() {
            return Err(format!(
                "package input {} is not a regular file",
                parent.path.join(&file_name).display()
            ));
        }
        let file = parent.directory.open(&file_name).map_err(|error| {
            format!(
                "cannot read package input {}: {error}",
                parent.path.join(&file_name).display()
            )
        })?;
        if !file
            .metadata()
            .map_err(|error| error.to_string())?
            .is_file()
        {
            return Err(format!(
                "package input {} is not a regular file",
                parent.path.join(&file_name).display()
            ));
        }
        Ok(file.into_std())
    }
}

fn open_application_cache_directory(
    project_root: &Path,
    cache_path: &Path,
) -> Result<PackageDirectory, String> {
    let entry = cache_path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "application cache directory has an invalid name".to_owned())?;
    let is_digest = entry.len() == 64 && entry.bytes().all(|byte| byte.is_ascii_hexdigit());
    let is_temporary = entry.strip_prefix(".extract-").is_some_and(|suffix| {
        !suffix.is_empty()
            && suffix
                .bytes()
                .all(|byte| byte.is_ascii_digit() || byte == b'-')
    });
    if !is_digest && !is_temporary {
        return Err("application cache directory has an invalid name".into());
    }
    let project = PackageDirectory::open_ambient(project_root)?;
    project
        .open_child_directory(Path::new("target"))?
        .open_child_directory(Path::new(".cog-application-cache"))?
        .open_child_directory(Path::new(entry))
}

/// A validated COGPKG 3 application extracted into the local package cache.
pub struct ApplicationPackage {
    /// Extracted application and dependency tree root.
    pub root: PathBuf,
    /// Bounded bytecode snapshot that passed metadata-aware validation.
    pub bytecode: Vec<u8>,
    /// Target-specific native plugins declared across the package graph.
    pub plugins: Vec<(PathBuf, crate::sdk::PluginManifest)>,
}

/// Safely extract and validate an application archive before execution.
pub fn open_application(path: &Path) -> Result<ApplicationPackage, String> {
    let root = extract_application_archive(path)?;
    let project_root = path.parent().unwrap_or(Path::new("."));
    let root_directory = open_application_cache_directory(project_root, &root)?;
    validate_application_package_directory(&root_directory)?;
    let lock_bytes = crate::limits::read_at_most(
        root_directory
            .open_regular_file(Path::new("cog.lock"))
            .map_err(|error| format!("cannot read application lockfile: {error}"))?,
        crate::config::MAX_CONFIGURATION_BYTES + 1,
    )
    .map_err(|error| format!("cannot read application lockfile: {error}"))?;
    if lock_bytes.len() > crate::config::MAX_CONFIGURATION_BYTES {
        return Err("application lockfile exceeds the 256 KiB limit".into());
    }
    let lock_text = std::str::from_utf8(&lock_bytes)
        .map_err(|error| format!("application lockfile is not UTF-8: {error}"))?;
    let lock = crate::lockfile::decode("cog.lock", lock_text)
        .map_err(|errors| crate::frontend::render(&errors))?;
    if lock
        .packages
        .first()
        .is_none_or(|package| package.path != ".")
    {
        return Err("application lockfile root package path is invalid".into());
    }
    let mut package_ids_by_path = BTreeMap::new();
    for (index, package) in lock.packages.iter().enumerate() {
        if index > 0 {
            validate_archive_path(&package.path)?;
        }
        if package_ids_by_path
            .insert(package.path.as_str(), index)
            .is_some()
        {
            return Err("application lockfile contains duplicate package paths".into());
        }
    }
    let lock_edges = lock
        .dependencies
        .iter()
        .cloned()
        .collect::<std::collections::BTreeSet<_>>();
    let mut manifest_edges = std::collections::BTreeSet::new();
    let mut plugins = Vec::new();
    let mut engine = crate::sdk::Engine::new();
    for (index, package) in lock.packages.iter().enumerate() {
        let relative = Path::new(&package.path);
        if index > 0 {
            validate_archive_path(&package.path)?;
        }
        let package_directory = if index == 0 {
            root_directory.try_clone()?
        } else {
            root_directory.open_child_directory(relative)?
        };
        let metadata = read_package_manifest(&package_directory)?;
        if metadata.name != package.name || metadata.version != package.version {
            return Err(format!(
                "application lockfile identity does not match package manifest at `{}`",
                package.path
            ));
        }
        if index == 0 && metadata.kind != config::PackageKind::Application {
            return Err("application package root must be an application".into());
        }
        if index > 0 && metadata.kind != config::PackageKind::Library {
            return Err("application package dependencies must be libraries".into());
        }
        for dependency in read_package_dependencies(&package_directory)? {
            if dependency.archive || dependency.installed.is_some() {
                return Err("COGPKG 3 dependencies must be vendored package directories".into());
            }
            let target_path = resolve_archive_dependency_path(&package.path, &dependency.path)?;
            let target = package_ids_by_path
                .get(target_path.as_str())
                .copied()
                .ok_or_else(|| {
                    format!(
                        "application dependency `{}` does not match a package in cog.lock",
                        dependency.alias
                    )
                })?;
            manifest_edges.insert((index, dependency.alias, target));
        }
        for plugin in metadata.plugins {
            for path in plugin.library_paths() {
                package_directory.open_regular_file(path)?;
            }
            plugin
                .register_metadata(&mut engine)
                .map_err(|error| error.to_string())?;
            plugins.push((package_directory.path.clone(), plugin));
        }
    }
    if manifest_edges != lock_edges {
        return Err("application dependency manifests do not match cog.lock".into());
    }
    let bytecode = crate::limits::read_at_most(
        root_directory
            .open_regular_file(Path::new(".cog/app.cogb"))
            .map_err(|error| format!("cannot read packaged bytecode: {error}"))?,
        crate::bytecode::MAX_FILE_BYTES + 1,
    )
    .map_err(|error| format!("cannot read packaged bytecode: {error}"))?;
    if bytecode.len() > crate::bytecode::MAX_FILE_BYTES {
        return Err(format!(
            "packaged bytecode exceeds the {}-byte limit",
            crate::bytecode::MAX_FILE_BYTES
        ));
    }
    engine
        .decode(&bytecode)
        .map_err(|error| format!("invalid packaged application bytecode: {error}"))?;
    Ok(ApplicationPackage {
        bytecode,
        root,
        plugins,
    })
}

fn read_package_manifest(root: &PackageDirectory) -> Result<Package, String> {
    let path = root.path.join("package.kdl");
    let bytes = crate::limits::read_at_most(
        root.open_regular_file(Path::new("package.kdl"))?,
        config::MAX_CONFIGURATION_BYTES + 1,
    )
    .map_err(|error| format!("cannot read application package manifest: {error}"))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("application package manifest is not UTF-8: {error}"))?;
    config::package_from_source(&path, text).map_err(|errors| crate::frontend::render(&errors))
}

fn read_package_dependencies(root: &PackageDirectory) -> Result<Vec<config::Dependency>, String> {
    let path = root.path.join("dependencies.kdl");
    let bytes = crate::limits::read_at_most(
        root.open_regular_file(Path::new("dependencies.kdl"))?,
        config::MAX_CONFIGURATION_BYTES + 1,
    )
    .map_err(|error| format!("cannot read application dependency manifest: {error}"))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|error| format!("application dependency manifest is not UTF-8: {error}"))?;
    config::dependencies_from_source(&path, text).map_err(|errors| crate::frontend::render(&errors))
}

fn resolve_archive_dependency_path(owner: &str, dependency: &Path) -> Result<String, String> {
    let dependency = dependency
        .to_str()
        .ok_or_else(|| "application dependency path is not UTF-8".to_owned())?;
    if dependency.is_empty()
        || dependency.starts_with('/')
        || dependency
            .chars()
            .any(|character| character.is_control() || matches!(character, '\\' | ':'))
    {
        return Err("application dependency path is not normalized".into());
    }
    let mut components = if owner == "." {
        Vec::new()
    } else {
        owner.split('/').map(str::to_owned).collect::<Vec<_>>()
    };
    for component in dependency.split('/') {
        match component {
            ".." => {
                if components.pop().is_none() {
                    return Err("application dependency path escapes the package root".into());
                }
            }
            "" | "." => return Err("application dependency path is not normalized".into()),
            normal => {
                validate_archive_path(normal)?;
                components.push(normal.to_owned());
            }
        }
    }
    if components.is_empty() {
        Ok(".".into())
    } else {
        Ok(components.join("/"))
    }
}

fn install_cache_directory(
    temporary: &Path,
    cache_path: &Path,
    validate: impl Fn(&Path) -> Result<(), String>,
) -> Result<(), String> {
    match std::fs::rename(temporary, cache_path) {
        Ok(()) => Ok(()),
        Err(error) => match validate(cache_path) {
            Ok(()) => Ok(()),
            Err(cache_error) => Err(format!(
                "cannot install package cache: {error}; existing cache is invalid: {cache_error}"
            )),
        },
    }
}

fn extract_archive_into(mut file: File, root: &Path) -> Result<(), String> {
    let archive_bytes = file
        .metadata()
        .map_err(|error| format!("cannot inspect package snapshot: {error}"))?
        .len();
    preflight_tar(&mut file, archive_bytes)?;
    file.seek(SeekFrom::Start(0))
        .map_err(|error| format!("cannot rewind package archive: {error}"))?;
    let mut has_manifest = false;

    {
        let mut archive = tar::Archive::new(&mut file);
        let entries = archive
            .entries()
            .map_err(|error| format!("invalid package TAR archive: {error}"))?;
        let mut seen = BTreeMap::<String, bool>::new();
        let mut folded_paths = BTreeMap::<String, String>::new();
        let mut entry_count = 0usize;
        let mut path_bytes = 0usize;
        let mut extracted_bytes = 0u64;

        for entry in entries {
            let mut entry = entry.map_err(|error| format!("invalid package TAR entry: {error}"))?;
            entry_count = entry_count
                .checked_add(1)
                .filter(|count| *count <= MAX_ARCHIVE_ENTRIES)
                .ok_or_else(|| {
                    format!("package archive entry limit ({MAX_ARCHIVE_ENTRIES}) exceeded")
                })?;
            let kind = entry.header().entry_type();
            let is_directory = kind == tar::EntryType::Directory;
            if !is_directory && kind != tar::EntryType::Regular {
                return Err(
                    "package archives may contain only regular files and directories".into(),
                );
            }
            let raw_path = entry
                .path()
                .map_err(|error| format!("invalid package archive path: {error}"))?;
            let raw_path = raw_path
                .to_str()
                .ok_or_else(|| "package archive paths must be valid UTF-8".to_owned())?;
            let path = if is_directory {
                raw_path.strip_suffix('/').unwrap_or(raw_path)
            } else {
                raw_path
            }
            .to_owned();
            validate_archive_path(&path)?;
            path_bytes = path_bytes
                .checked_add(path.len())
                .filter(|bytes| *bytes <= MAX_ARCHIVE_PATH_BYTES)
                .ok_or_else(|| {
                    format!("package archive path budget ({MAX_ARCHIVE_PATH_BYTES} bytes) exceeded")
                })?;
            record_case_folds(&path, &mut folded_paths)?;
            for parent in archive_parents(&path) {
                if seen.get(parent) == Some(&false) {
                    return Err(format!(
                        "package archive file `{parent}` conflicts with `{path}`"
                    ));
                }
            }
            let child_prefix = format!("{path}/");
            if !is_directory
                && seen
                    .range(child_prefix.clone()..)
                    .next()
                    .is_some_and(|(existing, _)| existing.starts_with(&child_prefix))
            {
                return Err(format!(
                    "package archive file `{path}` conflicts with a directory"
                ));
            }
            if seen.insert(path.clone(), is_directory).is_some() {
                return Err(format!("duplicate package archive path `{path}`"));
            }
            let destination = root.join(&path);
            if is_directory {
                if entry.size() != 0 {
                    return Err("package archive directory entries must be empty".into());
                }
                std::fs::create_dir_all(&destination).map_err(|error| {
                    format!("cannot create extracted package directory: {error}")
                })?;
                continue;
            }

            let size = entry.size();
            extracted_bytes = extracted_bytes
                .checked_add(size)
                .filter(|bytes| *bytes <= MAX_ARCHIVE_BYTES)
                .ok_or_else(|| {
                    format!("package archive extracted byte limit ({MAX_ARCHIVE_BYTES}) exceeded")
                })?;
            has_manifest |= path == "package.kdl";
            let parent = destination
                .parent()
                .ok_or_else(|| "invalid package archive destination".to_owned())?;
            std::fs::create_dir_all(parent)
                .map_err(|error| format!("cannot create extracted package directory: {error}"))?;
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&destination)
                .map_err(|error| format!("cannot create extracted package file: {error}"))?;
            let copied = io::copy(&mut entry, &mut output)
                .map_err(|error| format!("cannot extract package file `{path}`: {error}"))?;
            if copied != size {
                return Err(format!("truncated package archive entry `{path}`"));
            }
        }
        if !has_manifest {
            return Err("package archive has no root package.kdl".into());
        }
    }
    Ok(())
}

fn preflight_tar(file: &mut File, archive_bytes: u64) -> Result<(), String> {
    let mut offset = 0u64;
    let mut header_count = 0usize;
    let mut bytes = [0u8; 512];
    while offset < archive_bytes {
        if archive_bytes - offset < bytes.len() as u64 {
            return Err("truncated package TAR header".into());
        }
        file.seek(SeekFrom::Start(offset))
            .map_err(|error| format!("cannot inspect package TAR header: {error}"))?;
        file.read_exact(&mut bytes)
            .map_err(|error| format!("cannot read package TAR header: {error}"))?;
        offset += bytes.len() as u64;
        if bytes.iter().all(|byte| *byte == 0) {
            let mut trailing = [0u8; 64 * 1024];
            loop {
                let count = file
                    .read(&mut trailing)
                    .map_err(|error| format!("cannot read package TAR trailer: {error}"))?;
                if count == 0 {
                    return Ok(());
                }
                if trailing[..count].iter().any(|byte| *byte != 0) {
                    return Err("nonzero data follows the package TAR end marker".into());
                }
            }
        }
        header_count = header_count
            .checked_add(1)
            .filter(|count| *count <= MAX_ARCHIVE_HEADERS)
            .ok_or_else(|| {
                format!("package archive header limit ({MAX_ARCHIVE_HEADERS}) exceeded")
            })?;

        let mut header = tar::Header::new_gnu();
        header.as_mut_bytes().copy_from_slice(&bytes);
        let checksum = header
            .cksum()
            .map_err(|error| format!("invalid package TAR header checksum: {error}"))?;
        let calculated = bytes[..148]
            .iter()
            .chain(&bytes[156..])
            .fold(8 * 32u32, |sum, byte| sum + u32::from(*byte));
        if checksum != calculated {
            return Err("package TAR header checksum mismatch".into());
        }

        let kind = header.entry_type();
        let is_long_name = kind.is_gnu_longname();
        if !is_long_name && kind != tar::EntryType::Regular && kind != tar::EntryType::Directory {
            return Err("package archives may contain only regular files and directories".into());
        }
        let size = header
            .entry_size()
            .map_err(|error| format!("invalid package TAR entry size: {error}"))?;
        if is_long_name && size > MAX_ARCHIVE_ENTRY_PATH_BYTES as u64 + 1 {
            return Err("package TAR long-path metadata exceeds the path budget".into());
        }
        let padded = size
            .checked_add(511)
            .map(|size| size & !511)
            .ok_or_else(|| "package TAR entry size overflow".to_owned())?;
        offset = offset
            .checked_add(padded)
            .filter(|next| *next <= archive_bytes)
            .ok_or_else(|| "truncated package TAR entry data".to_owned())?;
    }
    Err("package TAR archive has no end marker".into())
}

pub struct BuildResult {
    pub path: PathBuf,
    pub entries: usize,
    pub archive_bytes: u64,
    pub warnings: Vec<crate::diagnostic::Diagnostic>,
}

/// Result of installing a locally built application or library archive.
pub struct InstallResult {
    pub path: PathBuf,
    pub package_name: String,
    pub package_version: String,
    pub kind: config::PackageKind,
    pub archive_bytes: u64,
    pub warnings: Vec<crate::diagnostic::Diagnostic>,
}

/// Build a project as a complete COGPKG archive and place it in the user store.
/// Existing package identities are immutable: installing different archive
/// contents under the same name and version is rejected.
pub fn install_path(project_path: &Path, locked: bool) -> Result<InstallResult, String> {
    let project_root = project_path.canonicalize().map_err(|error| {
        format!(
            "cannot resolve install path {}: {error}",
            project_path.display()
        )
    })?;
    if !project_root.is_dir() {
        return Err("cog install --path requires a project directory".into());
    }
    let package = config::package(&project_root.join("package.kdl"))
        .map_err(|errors| crate::frontend::render(&errors))?;
    let store_root = user_store_root()?;
    ensure_store_directory(&store_root)?;
    let category = store_root.join(match package.kind {
        config::PackageKind::Library => "libraries",
        config::PackageKind::Application => "programs",
    });
    ensure_store_directory(&category)?;
    let name_directory = category.join(store_component(&package.name));
    ensure_store_directory(&name_directory)?;
    let package_directory = name_directory.join(store_component(&package.version));
    ensure_store_directory(&package_directory)?;

    let stage = loop {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let candidate = package_directory.join(format!(".install-{}-{suffix}", std::process::id()));
        match std::fs::create_dir(&candidate) {
            Ok(()) => break candidate,
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => {
                return Err(format!(
                    "cannot create package install staging directory: {error}"
                ));
            }
        }
    };
    let staged_archive = stage.join("package.cogpkg");
    let destination = package_directory.join("package.cogpkg");
    let result = (|| {
        let built = build_archive(&project_root, Some(&staged_archive), locked)?;
        let current_package = config::package(&project_root.join("package.kdl"))
            .map_err(|errors| crate::frontend::render(&errors))?;
        if current_package.name != package.name
            || current_package.version != package.version
            || current_package.kind != package.kind
        {
            return Err(
                "package identity changed while the install archive was being built".into(),
            );
        }
        match std::fs::symlink_metadata(&destination) {
            Ok(metadata) if metadata.file_type().is_symlink() || !metadata.is_file() => {
                return Err(format!(
                    "installed package destination {} is not a regular file",
                    destination.display()
                ));
            }
            Ok(_) => {
                let existing_digest = package_archive_digest(&destination)?;
                let staged_digest = package_archive_digest(&staged_archive)?;
                if existing_digest != staged_digest {
                    return Err(format!(
                        "package `{}` version `{}` is already installed with different contents; bump the package version before installing",
                        package.name, package.version
                    ));
                }
            }
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                match std::fs::hard_link(&staged_archive, &destination) {
                    Ok(()) => {}
                    Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
                        if package_archive_digest(&destination)?
                            != package_archive_digest(&staged_archive)?
                        {
                            return Err(format!(
                                "package `{}` version `{}` is already installed with different contents; bump the package version before installing",
                                package.name, package.version
                            ));
                        }
                    }
                    Err(error) => {
                        return Err(format!(
                            "cannot install package archive {} atomically: {error}",
                            destination.display()
                        ));
                    }
                }
            }
            Err(error) => {
                return Err(format!(
                    "cannot inspect package install destination: {error}"
                ));
            }
        }
        Ok(InstallResult {
            path: destination,
            package_name: package.name.clone(),
            package_version: package.version.clone(),
            kind: package.kind,
            archive_bytes: built.archive_bytes,
            warnings: built.warnings,
        })
    })();
    match (result, std::fs::remove_dir_all(&stage)) {
        (Ok(installed), Ok(())) => Ok(installed),
        (Ok(installed), Err(error)) => Err(format!(
            "installed package at {}, but cannot remove staging directory: {error}",
            installed.path.display()
        )),
        (Err(error), Ok(())) => Err(error),
        (Err(error), Err(cleanup)) => Err(format!(
            "{error}; cannot remove package install staging directory: {cleanup}"
        )),
    }
}

fn user_store_root() -> Result<PathBuf, String> {
    if let Some(path) = std::env::var_os("COG_HOME") {
        let path = PathBuf::from(path);
        if !path.is_absolute() {
            return Err("COG_HOME must be an absolute path".into());
        }
        return Ok(path);
    }
    #[cfg(windows)]
    let home = std::env::var_os("USERPROFILE")
        .map(PathBuf::from)
        .or_else(|| {
            let mut path = std::env::var_os("HOMEDRIVE")?;
            path.push(std::env::var_os("HOMEPATH")?);
            Some(PathBuf::from(path))
        });
    #[cfg(not(windows))]
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let root = home
        .map(|path| path.join(".cog"))
        .ok_or_else(|| "cannot determine the user home directory; set COG_HOME".to_owned())?;
    if !root.is_absolute() {
        return Err("the user home directory must be absolute; set COG_HOME".into());
    }
    Ok(root)
}

fn store_component(value: &str) -> String {
    let display = safe_file_component(value);
    let digest = blake3::hash(value.as_bytes()).to_hex().to_string();
    format!("{display}-{}", &digest[..16])
}

fn ensure_store_directory(path: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(path) {
        Ok(metadata) if !metadata.file_type().is_symlink() && metadata.is_dir() => Ok(()),
        Ok(_) => Err(format!(
            "package store path {} is not a regular directory",
            path.display()
        )),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            std::fs::create_dir_all(path).map_err(|error| {
                format!(
                    "cannot create package store directory {}: {error}",
                    path.display()
                )
            })?;
            let metadata = std::fs::symlink_metadata(path).map_err(|error| {
                format!(
                    "cannot inspect package store directory {}: {error}",
                    path.display()
                )
            })?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(format!(
                    "package store path {} is not a regular directory",
                    path.display()
                ));
            }
            Ok(())
        }
        Err(error) => Err(format!(
            "cannot inspect package store directory {}: {error}",
            path.display()
        )),
    }
}

fn package_archive_digest(path: &Path) -> Result<[u8; 32], String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| format!("cannot inspect package archive {}: {error}", path.display()))?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(format!(
            "package archive {} is not a regular file",
            path.display()
        ));
    }
    if metadata.len() > MAX_COMPRESSED_ARCHIVE_BYTES {
        return Err("compressed package archive byte limit exceeded".into());
    }
    let mut file = File::open(path)
        .map_err(|error| format!("cannot read package archive {}: {error}", path.display()))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    let mut bytes = 0u64;
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| format!("cannot read package archive {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        bytes = bytes
            .checked_add(count as u64)
            .filter(|bytes| *bytes <= MAX_COMPRESSED_ARCHIVE_BYTES)
            .ok_or_else(|| "compressed package archive byte limit exceeded".to_owned())?;
        hasher.update(&buffer[..count]);
    }
    if bytes != metadata.len() {
        return Err(format!(
            "package archive {} changed while being read",
            path.display()
        ));
    }
    Ok(*hasher.finalize().as_bytes())
}

struct Entry {
    path: String,
    size: u64,
    source: EntrySource,
    source_path: Option<PathBuf>,
}

enum EntrySource {
    File { root: PathBuf, relative: PathBuf },
    Bytes(Vec<u8>),
}

#[derive(Default)]
struct ArchivePlan {
    entries: BTreeMap<String, Entry>,
    folded_paths: BTreeMap<String, String>,
    path_bytes: usize,
    data_bytes: u64,
}

impl ArchivePlan {
    fn add_file_snapshot(
        &mut self,
        archive_path: String,
        package_root: &Path,
        relative_path: &Path,
    ) -> Result<(), String> {
        let (source_path, bytes) = snapshot_package_file(package_root, relative_path)?;
        self.add_bytes(archive_path, bytes, Some(source_path))
    }

    fn add_file(
        &mut self,
        archive_path: String,
        package_root: &Path,
        relative_path: &Path,
    ) -> Result<(), String> {
        let relative_path = normalized_relative(relative_path)?;
        let source_path = package_root.join(&relative_path);
        if self.is_same_input(&archive_path, &source_path)? {
            return Ok(());
        }
        let file = open_regular_package_file(package_root, &relative_path)?;
        let size = file
            .metadata()
            .map_err(|error| format!("cannot inspect {}: {error}", source_path.display()))?
            .len();
        drop(file);
        self.insert(Entry {
            path: archive_path,
            size,
            source: EntrySource::File {
                root: package_root.to_path_buf(),
                relative: relative_path,
            },
            source_path: Some(source_path),
        })
    }

    fn add_bytes(
        &mut self,
        archive_path: String,
        bytes: Vec<u8>,
        source_path: Option<PathBuf>,
    ) -> Result<(), String> {
        if let Some(source_path) = &source_path
            && self.is_same_input(&archive_path, source_path)?
        {
            return Ok(());
        }
        let size = u64::try_from(bytes.len())
            .map_err(|_| "generated package entry is too large".to_owned())?;
        self.insert(Entry {
            path: archive_path,
            size,
            source: EntrySource::Bytes(bytes),
            source_path,
        })
    }

    fn is_same_input(&self, archive_path: &str, source_path: &Path) -> Result<bool, String> {
        let Some(existing) = self.entries.get(archive_path) else {
            return Ok(false);
        };
        if existing.source_path.as_deref() == Some(source_path) {
            Ok(true)
        } else {
            Err(format!("multiple package inputs map to `{archive_path}`"))
        }
    }

    fn insert(&mut self, entry: Entry) -> Result<(), String> {
        validate_archive_path(&entry.path)?;
        self.data_bytes = self
            .data_bytes
            .checked_add(entry.size)
            .filter(|bytes| *bytes <= MAX_ARCHIVE_BYTES)
            .ok_or_else(|| {
                format!("package archive content limit ({MAX_ARCHIVE_BYTES}) exceeded")
            })?;
        if self.entries.contains_key(&entry.path) {
            return Err(format!("duplicate package path `{}`", entry.path));
        }
        record_case_folds(&entry.path, &mut self.folded_paths)?;
        for parent in archive_parents(&entry.path) {
            if self.entries.contains_key(parent) {
                return Err(format!(
                    "package file `{parent}` conflicts with `{}`",
                    entry.path
                ));
            }
        }
        let child_prefix = format!("{}/", entry.path);
        if self
            .entries
            .range(child_prefix.clone()..)
            .next()
            .is_some_and(|(path, _)| path.starts_with(&child_prefix))
        {
            return Err(format!(
                "package file `{}` conflicts with an existing directory",
                entry.path
            ));
        }
        if self.entries.len() >= MAX_ARCHIVE_ENTRIES {
            return Err(format!(
                "package archive entry limit ({MAX_ARCHIVE_ENTRIES}) exceeded"
            ));
        }
        self.path_bytes = self
            .path_bytes
            .checked_add(entry.path.len())
            .filter(|bytes| *bytes <= MAX_ARCHIVE_PATH_BYTES)
            .ok_or_else(|| {
                format!("package archive path budget ({MAX_ARCHIVE_PATH_BYTES} bytes) exceeded")
            })?;
        self.entries.insert(entry.path.clone(), entry);
        Ok(())
    }

    fn source_paths(&self) -> impl Iterator<Item = &Path> {
        self.entries
            .values()
            .filter_map(|entry| entry.source_path.as_deref())
    }
}

/// Build a deterministic TAR+Zstandard package from a source project.
/// Dependencies are placed under `.cog/packages/<id>` and their local paths
/// and lockfile records are rewritten to remain inside the archive.
pub fn build_archive(
    project_path: &Path,
    output: Option<&Path>,
    locked: bool,
) -> Result<BuildResult, String> {
    build_archive_with_target(project_path, output, locked, None)
}

/// Build a package containing only the native plugin variant for `target`.
/// Cog bytecode remains portable; this selects packaged native libraries.
pub fn build_archive_for_target(
    project_path: &Path,
    output: Option<&Path>,
    locked: bool,
    target: &str,
) -> Result<BuildResult, String> {
    if !crate::sdk::supports_plugin_target(target) {
        return Err(format!("unsupported package target `{target}`"));
    }
    build_archive_with_target(project_path, output, locked, Some(target))
}

fn build_archive_with_target(
    project_path: &Path,
    output: Option<&Path>,
    locked: bool,
    target: Option<&str>,
) -> Result<BuildResult, String> {
    if !project_path.is_dir() {
        return Err("package build requires a project directory".into());
    }
    let selected_target = target
        .map(str::to_owned)
        .or_else(|| host_plugin_target().map(str::to_owned));
    let mut prepared = project::prepare(project_path, None, locked)
        .map_err(|errors| crate::frontend::render(&errors))?;
    crate::build::snapshot_sources(&mut prepared)?;
    let roots = prepared.package_roots.clone();
    if roots.first() != Some(&prepared.root) {
        return Err("internal error: resolved package graph has no root".into());
    }
    let packages = prepared.package_metadata.clone();
    let root_package = packages
        .first()
        .ok_or_else(|| "internal error: resolved package graph is empty".to_owned())?;
    if root_package.kind != prepared.kind {
        return Err("package metadata changed while preparing the archive".into());
    }

    let output = output.map(Path::to_path_buf).unwrap_or_else(|| {
        let name = safe_file_component(&root_package.name);
        let file_name = target.map_or_else(
            || format!("package-{name}.cogpkg"),
            |target| format!("package-{name}-{target}.cogpkg"),
        );
        prepared.root.join("target").join(file_name)
    });
    if let Ok(metadata) = std::fs::symlink_metadata(&output) {
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "package output {} is a symbolic link",
                output.display()
            ));
        }
        if !metadata.is_file() {
            return Err(format!("package output {} is not a file", output.display()));
        }
    }
    let output = crate::build::resolve_build_path(&output)?;
    let mut plan = package_plan(&prepared, &packages, selected_target.as_deref())?;
    protect_archive_output(&prepared, &plan, &output)?;
    let build_inputs = prepared.build_inputs.clone();

    let mut engine = crate::sdk::Engine::new();
    for (_, manifest) in &prepared.plugins {
        manifest
            .register_metadata(&mut engine)
            .map_err(|error| error.to_string())?;
    }
    let warnings = if prepared.kind == config::PackageKind::Library {
        engine
            .check_library_prepared(prepared)
            .map_err(|error| error.to_string())?
    } else {
        let (program, warnings) = engine
            .compile_prepared_with_warnings(prepared)
            .map_err(|error| error.to_string())?;
        plan.add_bytes(
            ".cog/app.cogb".into(),
            program.encode().map_err(|error| error.to_string())?,
            None,
        )?;
        warnings
    };

    crate::build::protect_build_inputs(&build_inputs, &output)?;
    let entries = plan.entries.len();
    let archive_bytes = write_archive(&output, std::mem::take(&mut plan.entries))?;
    Ok(BuildResult {
        path: output,
        entries,
        archive_bytes,
        warnings,
    })
}

fn package_plan(
    prepared: &Prepared,
    packages: &[Package],
    target: Option<&str>,
) -> Result<ArchivePlan, String> {
    let mut plan = ArchivePlan::default();
    let marker: &[u8] = if prepared.kind == config::PackageKind::Application {
        b"COGPKG 3\n"
    } else {
        b"COGPKG 2\n"
    };
    plan.add_bytes("COGPKG".into(), marker.to_vec(), None)?;
    for manifest in &prepared.manifests {
        ensure_no_symlink_manifest(&prepared.package_roots, manifest)?;
    }

    for (package_id, (root, metadata)) in prepared.package_roots.iter().zip(packages).enumerate() {
        let manifest_path = Path::new("package.kdl");
        let (manifest_source, manifest_bytes) = snapshot_package_file(root, manifest_path)?;
        let manifest_text = std::str::from_utf8(&manifest_bytes)
            .map_err(|error| format!("package manifest is not UTF-8: {error}"))?;
        let current = config::package_from_source(&root.join(manifest_path), manifest_text)
            .map_err(|errors| crate::frontend::render(&errors))?;
        if !same_package_metadata(&current, metadata) {
            return Err(format!(
                "package manifest {} changed after project preparation",
                root.join(manifest_path).display()
            ));
        }
        let package_manifest = filter_package_manifest_target(manifest_text, target)?;
        plan.add_bytes(
            package_archive_path(package_id, manifest_path)?,
            package_manifest,
            Some(manifest_source),
        )?;
        let dependencies = package_dependencies(prepared, package_id)?;
        plan.add_bytes(
            package_archive_path(package_id, Path::new("dependencies.kdl"))?,
            dependencies.into_bytes(),
            None,
        )?;

        for asset in &metadata.assets {
            plan.add_file_snapshot(package_archive_path(package_id, asset)?, root, asset)?;
        }
        for plugin in &metadata.plugins {
            let paths: Vec<&Path> = if plugin.target_libraries.is_empty() {
                vec![plugin.library.as_path()]
            } else {
                let target = target.ok_or_else(|| {
                    "a target-specific native plugin requires --target on this host".to_owned()
                })?;
                let path = plugin.target_libraries.get(target).ok_or_else(|| {
                    format!("package plugin does not declare a library for target `{target}`")
                })?;
                vec![path.as_path()]
            };
            let mut added = std::collections::BTreeSet::new();
            for path in paths {
                if added.insert(path.to_path_buf()) {
                    plan.add_file_snapshot(package_archive_path(package_id, path)?, root, path)?;
                }
            }
        }
        if package_id == 0
            && let Some(build_manifest) = &prepared.build_manifest
        {
            plan.add_bytes(
                "build.kdl".into(),
                build_manifest.as_bytes().to_vec(),
                Some(root.join("build.kdl")),
            )?;
        }
    }

    for (key, source) in &prepared.sources {
        match source {
            Source::File(path) => {
                let root = package_root(prepared, key.0)?;
                let relative = path.strip_prefix(root).map_err(|_| {
                    format!("source {} is outside its package root", path.display())
                })?;
                let archive_path = package_archive_path(key.0, relative)?;
                plan.add_file(archive_path, root, relative)?;
            }
            Source::Snapshot(path, bytes) => {
                let root = package_root(prepared, key.0)?;
                let relative = path.strip_prefix(root).map_err(|_| {
                    format!("source {} is outside its package root", path.display())
                })?;
                plan.add_bytes(
                    package_archive_path(key.0, relative)?,
                    bytes.clone(),
                    Some(path.clone()),
                )?;
            }
            Source::Embedded(_, _) | Source::Empty => {}
        }
    }

    let lock = crate::lockfile::Lockfile {
        packages: packages
            .iter()
            .enumerate()
            .map(|(id, package)| crate::lockfile::Package {
                name: package.name.clone(),
                version: package.version.clone(),
                path: package_archive_package_root(id),
            })
            .collect(),
        dependencies: prepared
            .aliases
            .iter()
            .filter_map(|((owner, alias), target)| {
                (*target != crate::resolver::CORE).then_some((*owner, alias.clone(), *target))
            })
            .collect(),
    };
    let lock = lock.encode()?;
    plan.add_bytes("cog.lock".into(), lock.into_bytes(), None)?;
    Ok(plan)
}

fn filter_package_manifest_target(manifest: &str, target: Option<&str>) -> Result<Vec<u8>, String> {
    let mut document = kdl::KdlDocument::parse_v2(manifest)
        .map_err(|error| format!("cannot filter package plugin targets: {error}"))?;
    let mut changed = false;
    for package in document
        .nodes_mut()
        .iter_mut()
        .filter(|node| node.name().value() == "package")
    {
        let Some(package_children) = package.children_mut().as_mut() else {
            continue;
        };
        for plugin in package_children
            .nodes_mut()
            .iter_mut()
            .filter(|node| node.name().value() == "plugin")
        {
            let Some(plugin_children) = plugin.children_mut().as_mut() else {
                continue;
            };
            if !plugin_children
                .nodes()
                .iter()
                .any(|node| node.name().value() == "library")
            {
                continue;
            }
            let target = target.ok_or_else(|| {
                "a target-specific native plugin requires --target on this host".to_owned()
            })?;
            let mut found = false;
            plugin_children.nodes_mut().retain(|node| {
                if node.name().value() != "library" {
                    return true;
                }
                let keep = node.get("target").and_then(|value| value.as_string()) == Some(target);
                found |= keep;
                keep
            });
            if !found {
                return Err(format!(
                    "package plugin does not declare a library for target `{target}`"
                ));
            }
            changed = true;
        }
    }
    if changed {
        Ok(document.to_string().into_bytes())
    } else {
        Ok(manifest.as_bytes().to_vec())
    }
}

fn host_plugin_target() -> Option<&'static str> {
    #[cfg(all(target_arch = "x86_64", target_os = "linux"))]
    return Some("x86_64-unknown-linux-gnu");
    #[cfg(all(target_arch = "x86_64", target_os = "windows"))]
    return Some("x86_64-pc-windows-msvc");
    #[cfg(all(target_arch = "aarch64", target_os = "macos"))]
    return Some("aarch64-apple-darwin");
    #[allow(unreachable_code)]
    None
}

fn snapshot_package_file(root: &Path, relative: &Path) -> Result<(PathBuf, Vec<u8>), String> {
    let relative = normalized_relative(relative)?;
    let source = root.join(&relative);
    let file = open_regular_package_file(root, &relative)?;
    let length = file
        .metadata()
        .map_err(|error| format!("cannot inspect {}: {error}", source.display()))?
        .len();
    if length > MAX_ARCHIVE_BYTES {
        return Err("package input exceeds archive content limit".into());
    }
    let max = usize::try_from(length)
        .ok()
        .and_then(|length| length.checked_add(1))
        .ok_or("package input is too large to snapshot")?;
    let bytes = crate::limits::read_at_most(file, max)
        .map_err(|error| format!("cannot snapshot {}: {error}", source.display()))?;
    if bytes.len() as u64 != length {
        return Err(format!(
            "package input {} changed while being read",
            source.display()
        ));
    }
    Ok((source, bytes))
}

fn same_package_metadata(left: &Package, right: &Package) -> bool {
    left.name == right.name
        && left.version == right.version
        && left.kind == right.kind
        && left.assets == right.assets
        && left.plugins.len() == right.plugins.len()
        && left
            .plugins
            .iter()
            .zip(&right.plugins)
            .all(|(left, right)| {
                left.library == right.library
                    && left.target_libraries == right.target_libraries
                    && left.functions().len() == right.functions().len()
                    && left.functions().iter().zip(right.functions()).all(
                        |((left_name, left_sig), (right_name, right_sig))| {
                            left_name == right_name
                                && left_sig.parameters == right_sig.parameters
                                && left_sig.result == right_sig.result
                                && left_sig.effects == right_sig.effects
                                && left_sig.mutating_parameters == right_sig.mutating_parameters
                        },
                    )
            })
}

fn package_dependencies(prepared: &Prepared, owner: usize) -> Result<String, String> {
    let mut output = String::new();
    for ((package, alias), target) in &prepared.aliases {
        if *package != owner || *target == crate::resolver::CORE {
            continue;
        }
        let path = relative_package_path(owner, *target)?;
        output.push_str(alias);
        output.push_str(" path=\"");
        output.push_str(&path);
        output.push_str("\"\n");
    }
    Ok(output)
}

fn relative_package_path(from: usize, to: usize) -> Result<String, String> {
    if from == to {
        return Err("package cannot depend on itself".into());
    }
    let from_parts = package_archive_components(from);
    let to_parts = package_archive_components(to);
    let common = from_parts
        .iter()
        .zip(&to_parts)
        .take_while(|(left, right)| left == right)
        .count();
    let mut parts = vec!["..".to_owned(); from_parts.len().saturating_sub(common)];
    parts.extend(to_parts[common..].iter().cloned());
    if parts.is_empty() {
        return Err("package dependency path resolves to its owner".into());
    }
    Ok(parts.join("/"))
}

fn package_archive_package_root(package_id: usize) -> String {
    if package_id == 0 {
        ".".into()
    } else {
        format!(".cog/packages/{package_id}")
    }
}

fn package_archive_components(package_id: usize) -> Vec<String> {
    if package_id == 0 {
        Vec::new()
    } else {
        vec![".cog".into(), "packages".into(), package_id.to_string()]
    }
}

fn package_archive_path(package_id: usize, relative: &Path) -> Result<String, String> {
    let relative = normalized_relative(relative)?;
    let mut parts = package_archive_components(package_id);
    for component in relative.components() {
        match component {
            Component::Normal(part) => parts.push(
                part.to_str()
                    .ok_or_else(|| "package paths must be valid UTF-8".to_owned())?
                    .to_owned(),
            ),
            _ => return Err("package paths must be normalized relative paths".into()),
        }
    }
    if package_id == 0 && parts.len() >= 2 && parts[0] == ".cog" && parts[1] == "packages" {
        return Err("the `.cog/packages` directory is reserved inside package archives".into());
    }
    validate_archive_path(&parts.join("/"))?;
    Ok(parts.join("/"))
}

fn normalized_relative(path: &Path) -> Result<PathBuf, String> {
    if path.is_absolute() {
        return Err("package input path must be relative".into());
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(part) => normalized.push(part),
            _ => return Err("package input path must be normalized and relative".into()),
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err("package input path must not be empty".into());
    }
    Ok(normalized)
}

fn validate_archive_path(path: &str) -> Result<(), String> {
    if path.is_empty()
        || path.len() > MAX_ARCHIVE_ENTRY_PATH_BYTES
        || path
            .chars()
            .any(|character| character.is_control() || matches!(character, '\\' | ':'))
        || path.split('/').any(|part| {
            part.is_empty()
                || matches!(part, "." | "..")
                || part.ends_with(['.', ' '])
                || is_reserved_windows_component(part)
        })
    {
        return Err(format!("unsafe package archive path `{path}`"));
    }
    Ok(())
}

fn record_case_folds(
    path: &str,
    folded_paths: &mut BTreeMap<String, String>,
) -> Result<(), String> {
    let mut prefix = String::new();
    for component in path.split('/') {
        if !prefix.is_empty() {
            prefix.push('/');
        }
        prefix.push_str(component);
        let folded = prefix.to_lowercase();
        if let Some(previous) = folded_paths.get(&folded) {
            if previous != &prefix {
                return Err(format!(
                    "package paths `{previous}` and `{prefix}` collide on case-insensitive filesystems"
                ));
            }
        } else {
            folded_paths.insert(folded, prefix.clone());
        }
    }
    Ok(())
}

fn is_reserved_windows_component(component: &str) -> bool {
    let stem = component
        .split('.')
        .next()
        .unwrap_or("")
        .to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (stem.len() == 4
            && matches!(&stem[..3], "COM" | "LPT")
            && matches!(stem.as_bytes()[3], b'1'..=b'9'))
}

fn archive_parents(path: &str) -> impl Iterator<Item = &str> {
    path.match_indices('/').map(|(index, _)| &path[..index])
}

fn package_root(prepared: &Prepared, package_id: usize) -> Result<&Path, String> {
    prepared
        .package_roots
        .get(package_id)
        .map(PathBuf::as_path)
        .ok_or_else(|| format!("internal error: unknown package ID {package_id}"))
}

fn open_regular_package_file(root: &Path, relative: &Path) -> Result<File, String> {
    let relative = normalized_relative(relative)?;
    let components = relative.components().collect::<Vec<_>>();
    let directory = cap_std::fs::Dir::open_ambient_dir(root, cap_std::ambient_authority())
        .map_err(|error| format!("cannot open package root {}: {error}", root.display()))?;
    let mut prefix = PathBuf::new();
    for (index, component) in components.iter().enumerate() {
        let Component::Normal(name) = component else {
            return Err("package input path must be normalized".into());
        };
        prefix.push(name);
        let metadata = directory.symlink_metadata(&prefix).map_err(|error| {
            format!("cannot inspect package input {}: {error}", prefix.display())
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "package input {} is a symbolic link",
                prefix.display()
            ));
        }
        let last = index + 1 == components.len();
        if (last && !metadata.is_file()) || (!last && !metadata.is_dir()) {
            return Err(format!(
                "package input {} is not a regular file",
                prefix.display()
            ));
        }
    }
    let file = directory
        .open(&relative)
        .map_err(|error| format!("cannot read package input {}: {error}", relative.display()))?;
    if !file
        .metadata()
        .map_err(|error| {
            format!(
                "cannot inspect package input {}: {error}",
                relative.display()
            )
        })?
        .is_file()
    {
        return Err(format!(
            "package input {} is not a regular file",
            relative.display()
        ));
    }
    Ok(file.into_std())
}

fn ensure_no_symlink_manifest(roots: &[PathBuf], manifest: &Path) -> Result<(), String> {
    let root = roots
        .iter()
        .find(|root| manifest.starts_with(root))
        .ok_or_else(|| {
            format!(
                "manifest {} is outside the package graph",
                manifest.display()
            )
        })?;
    let relative = manifest
        .strip_prefix(root)
        .map_err(|_| "internal error: manifest root mismatch".to_owned())?;
    let relative = normalized_relative(relative)?;
    let mut path = root.clone();
    for component in relative.components() {
        let Component::Normal(name) = component else {
            return Err("manifest path is not normalized".into());
        };
        path.push(name);
        match std::fs::symlink_metadata(&path) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                return Err(format!(
                    "package manifest {} is a symbolic link",
                    path.display()
                ));
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(()),
            Err(error) => {
                return Err(format!(
                    "cannot inspect package manifest {}: {error}",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

fn protect_archive_output(
    prepared: &Prepared,
    plan: &ArchivePlan,
    output: &Path,
) -> Result<(), String> {
    for input in plan
        .source_paths()
        .chain(prepared.archive_inputs.iter().map(PathBuf::as_path))
    {
        let input = crate::build::resolve_build_path(input)?;
        if crate::build::same_build_path(&input, output) {
            return Err(format!(
                "package output would overwrite a project input: {}",
                output.display()
            ));
        }
    }
    let lock = prepared.root.join("cog.lock");
    for input in prepared.manifests.iter().chain(std::iter::once(&lock)) {
        let input = crate::build::resolve_build_path(input)?;
        let configuration_alias =
            input
                .file_name()
                .zip(output.file_name())
                .is_some_and(|(left, right)| {
                    left.as_encoded_bytes()
                        .eq_ignore_ascii_case(right.as_encoded_bytes())
                })
                && input
                    .parent()
                    .zip(output.parent())
                    .is_some_and(|(left, right)| crate::build::same_build_path(left, right));
        if configuration_alias || crate::build::same_build_path(&input, output) {
            return Err(format!(
                "package output would overwrite a project input: {}",
                output.display()
            ));
        }
    }
    Ok(())
}

fn safe_file_component(value: &str) -> String {
    let component = value
        .chars()
        .take(100)
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_' | '.') {
                character
            } else {
                '-'
            }
        })
        .collect::<String>();
    let component = component.trim_matches(['.', ' ']);
    if component.is_empty() {
        "package".into()
    } else {
        component.into()
    }
}

fn write_archive(path: &Path, entries: BTreeMap<String, Entry>) -> Result<u64, String> {
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty());
    if let Some(parent) = parent {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
    let temporary = path.with_extension(format!("cogpkg.tmp.{}.{suffix}", std::process::id()));
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|error| format!("cannot create {}: {error}", temporary.display()))?;

    let result = (|| {
        let compressed = CountingWriter::new(file, MAX_COMPRESSED_ARCHIVE_BYTES);
        let mut encoder = zstd::stream::write::Encoder::new(compressed, ZSTD_COMPRESSION_LEVEL)
            .map_err(|error| format!("cannot initialize Zstandard package encoder: {error}"))?;
        encoder
            .include_checksum(true)
            .map_err(|error| format!("cannot configure Zstandard package checksum: {error}"))?;
        let mut archive = tar::Builder::new(CountingWriter::new(encoder, MAX_ARCHIVE_BYTES));
        archive.follow_symlinks(false);
        for entry in entries.into_values() {
            let mut header = tar::Header::new_gnu();
            header.set_entry_type(tar::EntryType::Regular);
            header.set_mode(0o644);
            header.set_uid(0);
            header.set_gid(0);
            header.set_mtime(0);
            header.set_size(entry.size);
            let result = match entry.source {
                EntrySource::File { root, relative } => {
                    let file = open_regular_package_file(&root, &relative)?;
                    let current_size = file
                        .metadata()
                        .map_err(|error| {
                            format!("cannot inspect package input {}: {error}", entry.path)
                        })?
                        .len();
                    if current_size != entry.size {
                        return Err(format!(
                            "package input `{}` changed size while the archive was being built",
                            entry.path
                        ));
                    }
                    archive.append_data(&mut header, &entry.path, file)
                }
                EntrySource::Bytes(bytes) => {
                    archive.append_data(&mut header, &entry.path, Cursor::new(bytes))
                }
            };
            result.map_err(|error| format!("cannot archive {}: {error}", entry.path))?;
        }
        let tar_writer = archive
            .into_inner()
            .map_err(|error| format!("cannot finish package archive: {error}"))?;
        if tar_writer.bytes == 0 {
            return Err("cannot write an empty package TAR".into());
        }
        let encoder = tar_writer.inner;
        let compressed_writer = encoder
            .finish()
            .map_err(|error| format!("cannot finish Zstandard package frame: {error}"))?;
        compressed_writer
            .inner
            .sync_all()
            .map_err(|error| format!("cannot sync {}: {error}", temporary.display()))?;
        Ok(compressed_writer.bytes)
    })();
    match result {
        Ok(bytes) => {
            replace_output(&temporary, path)?;
            Ok(bytes)
        }
        Err(error) => {
            let _ = std::fs::remove_file(&temporary);
            Err(error)
        }
    }
}

struct CountingWriter<W> {
    inner: W,
    bytes: u64,
    limit: u64,
}

impl<W> CountingWriter<W> {
    fn new(inner: W, limit: u64) -> Self {
        Self {
            inner,
            bytes: 0,
            limit,
        }
    }
}

impl<W: Write> Write for CountingWriter<W> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        if self
            .bytes
            .checked_add(buffer.len() as u64)
            .is_none_or(|bytes| bytes > self.limit)
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "package archive byte limit exceeded",
            ));
        }
        let written = self.inner.write(buffer)?;
        self.bytes = self
            .bytes
            .checked_add(written as u64)
            .ok_or_else(|| io::Error::other("package archive byte count overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn replace_output(temporary: &Path, output: &Path) -> Result<(), String> {
    #[cfg(not(windows))]
    {
        std::fs::rename(temporary, output).map_err(|error| {
            let _ = std::fs::remove_file(temporary);
            format!("cannot write package archive {}: {error}", output.display())
        })
    }
    #[cfg(windows)]
    {
        if !output.exists() {
            return std::fs::rename(temporary, output).map_err(|error| {
                let _ = std::fs::remove_file(temporary);
                format!("cannot write package archive {}: {error}", output.display())
            });
        }
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let backup = output.with_extension(format!("cogpkg.old.{}.{suffix}", std::process::id()));
        std::fs::rename(output, &backup).map_err(|error| {
            let _ = std::fs::remove_file(temporary);
            format!(
                "cannot preserve existing package archive {}: {error}",
                output.display()
            )
        })?;
        match std::fs::rename(temporary, output) {
            Ok(()) => {
                std::fs::remove_file(&backup).map_err(|error| {
                    format!(
                        "package archive created but cannot remove {}: {error}",
                        backup.display()
                    )
                })?;
                Ok(())
            }
            Err(error) => {
                if let Err(restore) = std::fs::rename(&backup, output) {
                    return Err(format!(
                        "cannot write package archive {}: {error}; previous output remains at {} because restoration failed: {restore}",
                        output.display(),
                        backup.display()
                    ));
                }
                let _ = std::fs::remove_file(temporary);
                Err(format!(
                    "cannot write package archive {}: {error}",
                    output.display()
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArchivePlan, CountingWriter, install_cache_directory, package_archive_path, preflight_tar,
        relative_package_path, validate_archive_path, validate_cached_package,
    };
    #[cfg(unix)]
    use super::{open_application_cache_directory, read_package_manifest};
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::path::Path;
    use std::sync::atomic::Ordering;

    #[test]
    fn archive_paths_are_relative_normalized_and_portable() {
        assert!(validate_archive_path("src/lib.cog").is_ok());
        for path in [
            "",
            "/absolute",
            "../escape",
            "a/../b",
            "a\\b",
            "C:/file",
            "a//b",
            "file.",
            "CON.txt",
            "COM1.bin",
        ] {
            assert!(validate_archive_path(path).is_err(), "accepted {path:?}");
        }
        assert!(package_archive_path(1, Path::new("src/lib.cog")).is_ok());
        assert!(package_archive_path(0, Path::new("../outside.cog")).is_err());
    }

    #[test]
    fn internal_dependency_paths_are_relative_to_each_package_root() {
        assert_eq!(relative_package_path(0, 1).unwrap(), ".cog/packages/1");
        assert_eq!(relative_package_path(1, 2).unwrap(), "../2");
        assert_eq!(relative_package_path(2, 1).unwrap(), "../1");
    }

    #[test]
    fn archive_rejects_case_collisions_in_parent_components() {
        let mut plan = ArchivePlan::default();
        plan.add_bytes("Foo/a.cog".into(), Vec::new(), None)
            .unwrap();
        let error = plan
            .add_bytes("foo/b.cog".into(), Vec::new(), None)
            .unwrap_err();
        assert!(error.contains("case-insensitive filesystems"), "{error}");

        let mut folds = std::collections::BTreeMap::new();
        super::record_case_folds("Foo/a.cog", &mut folds).unwrap();
        assert!(super::record_case_folds("foo/b.cog", &mut folds).is_err());
    }

    #[test]
    fn archive_writer_enforces_its_output_byte_limit() {
        let mut writer = CountingWriter::new(Vec::new(), 2);
        assert!(writer.write_all(b"123").is_err());
        assert!(writer.inner.is_empty());
    }

    #[test]
    fn zstd_package_decoder_rejects_trailing_data_and_concatenated_frames() {
        let root = std::env::temp_dir().join(format!(
            "cog-package-zstd-trailing-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let frame = zstd::stream::encode_all(&b"payload"[..], 6).unwrap();
        let second = zstd::stream::encode_all(&b"second"[..], 6).unwrap();
        for (name, suffix) in [
            ("trailing", &b"junk"[..]),
            ("second-frame", second.as_slice()),
        ] {
            let input = root.join(format!("{name}.cogpkg"));
            let mut contents = frame.clone();
            contents.extend_from_slice(suffix);
            std::fs::write(&input, contents).unwrap();
            let error =
                super::decompress_archive(std::fs::File::open(&input).unwrap(), &root).unwrap_err();
            assert!(error.contains("data follows"), "{error}");
            assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);
            std::fs::remove_file(input).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn zstd_package_decoder_rejects_truncation_and_expansion_over_limit() {
        let root = std::env::temp_dir().join(format!(
            "cog-package-zstd-limits-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let frame = zstd::stream::encode_all(&b"expanded package payload"[..], 6).unwrap();
        let truncated = root.join("truncated.cogpkg");
        std::fs::write(&truncated, &frame[..frame.len() - 2]).unwrap();
        assert!(
            super::decompress_archive(std::fs::File::open(&truncated).unwrap(), &root).is_err()
        );
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 1);

        let bounded = root.join("bounded.cogpkg");
        std::fs::write(&bounded, &frame).unwrap();
        let error =
            super::decompress_archive_with_limit(std::fs::File::open(&bounded).unwrap(), &root, 8)
                .unwrap_err();
        assert!(error.contains("byte limit"), "{error}");
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 2);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn tar_preflight_bounds_long_name_extensions_before_entry_parsing() {
        let path = std::env::temp_dir().join(format!(
            "cog-long-tar-name-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let file = std::fs::File::create_new(&path).unwrap();
        let mut archive = tar::Builder::new(file);
        let long_path = format!("{}/{}.cog", "a".repeat(150), "b".repeat(150));
        for (entry_path, contents) in [
            ("COGPKG", &b"COGPKG 1\n"[..]),
            ("package.kdl", &b"package \"long-path\" version=\"1\"\n"[..]),
            (long_path.as_str(), &b"source"[..]),
        ] {
            let mut header = tar::Header::new_gnu();
            header.set_size(contents.len() as u64);
            header.set_cksum();
            archive
                .append_data(&mut header, entry_path, contents)
                .unwrap();
        }
        archive.finish().unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let archive_bytes = file.metadata().unwrap().len();
        super::preflight_tar(&mut file, archive_bytes).unwrap();
        let extracted = std::env::temp_dir().join(format!(
            "cog-long-tar-extracted-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&extracted).unwrap();
        super::extract_archive_into(std::fs::File::open(&path).unwrap(), &extracted).unwrap();
        assert_eq!(std::fs::read(extracted.join(long_path)).unwrap(), b"source");
        std::fs::remove_dir_all(extracted).unwrap();
        std::fs::remove_file(&path).unwrap();

        let path = std::env::temp_dir().join(format!(
            "cog-long-tar-limit-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let file = std::fs::File::create_new(&path).unwrap();
        let mut archive = tar::Builder::new(file);
        let mut header = tar::Header::new_gnu();
        header.set_entry_type(tar::EntryType::GNULongName);
        header.set_path("././@LongLink").unwrap();
        header.set_size((super::MAX_ARCHIVE_ENTRY_PATH_BYTES + 2) as u64);
        header.set_cksum();
        let oversized = vec![b'a'; super::MAX_ARCHIVE_ENTRY_PATH_BYTES + 2];
        archive.append(&header, oversized.as_slice()).unwrap();
        archive.finish().unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let archive_bytes = file.metadata().unwrap().len();
        assert!(
            super::preflight_tar(&mut file, archive_bytes)
                .unwrap_err()
                .contains("long-path metadata")
        );
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn concurrent_cache_publication_accepts_an_existing_valid_package() {
        let root = std::env::temp_dir().join(format!(
            "cog-package-cache-race-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let temporary = root.join(".extract");
        let cache = root.join("digest");
        for (path, name) in [(&temporary, "temporary"), (&cache, "published")] {
            std::fs::create_dir(path).unwrap();
            std::fs::write(path.join("COGPKG"), b"COGPKG 2\n").unwrap();
            std::fs::write(
                path.join("package.kdl"),
                format!("package \"{name}\" version=\"1\" kind=\"library\"\n"),
            )
            .unwrap();
        }

        install_cache_directory(&temporary, &cache, validate_cached_package).unwrap();
        assert!(
            temporary.is_dir(),
            "the concurrent winner owns the cache path"
        );
        assert_eq!(
            std::fs::read_to_string(cache.join("package.kdl")).unwrap(),
            "package \"published\" version=\"1\" kind=\"library\"\n"
        );
        std::fs::remove_dir_all(root).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn pinned_application_package_directory_survives_cache_path_replacement() {
        let project = std::env::temp_dir().join(format!(
            "cog-package-pinned-cache-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let digest = "a".repeat(64);
        let cache = project.join("target/.cog-application-cache").join(&digest);
        let vendored = cache.join(".cog/packages/dep");
        std::fs::create_dir_all(&vendored).unwrap();
        std::fs::write(
            vendored.join("package.kdl"),
            "package \"pinned-lib\" version=\"0.1.0\" kind=\"library\"\n",
        )
        .unwrap();

        let root = open_application_cache_directory(&project, &cache).unwrap();
        let package = root
            .open_child_directory(Path::new(".cog/packages/dep"))
            .unwrap();

        let packages = cache.join(".cog/packages");
        let moved_packages = cache.join(".cog/packages-original");
        std::fs::rename(&packages, &moved_packages).unwrap();
        let outside = project.join("outside-packages");
        std::fs::create_dir(&outside).unwrap();
        std::fs::write(
            outside.join("package.kdl"),
            "package \"outside-lib\" version=\"0.1.0\" kind=\"library\"\n",
        )
        .unwrap();
        std::os::unix::fs::symlink(&outside, &packages).unwrap();

        let manifest = read_package_manifest(&package).unwrap();
        assert_eq!(manifest.name, "pinned-lib");

        drop(package);
        drop(root);
        std::fs::remove_dir_all(project).unwrap();
    }

    #[test]
    fn tar_preflight_rejects_truncated_entry_data_and_bad_header_checksum() {
        let path = std::env::temp_dir().join(format!(
            "cog-truncated-tar-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let file = std::fs::File::create_new(&path).unwrap();
        let mut archive = tar::Builder::new(file);
        let mut header = tar::Header::new_gnu();
        header.set_size(4);
        header.set_cksum();
        archive
            .append_data(&mut header, "COGPKG", &b"data"[..])
            .unwrap();
        archive.finish().unwrap();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(512 + 100).unwrap();
        let mut file = std::fs::File::open(&path).unwrap();
        let archive_bytes = file.metadata().unwrap().len();
        assert!(
            preflight_tar(&mut file, archive_bytes)
                .unwrap_err()
                .contains("truncated package TAR entry data")
        );
        std::fs::remove_file(&path).unwrap();

        let path = std::env::temp_dir().join(format!(
            "cog-corrupt-tar-{}-{}",
            std::process::id(),
            super::NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let file = std::fs::File::create_new(&path).unwrap();
        let mut archive = tar::Builder::new(file);
        let mut header = tar::Header::new_gnu();
        header.set_size(4);
        header.set_cksum();
        archive
            .append_data(&mut header, "COGPKG", &b"data"[..])
            .unwrap();
        archive.finish().unwrap();
        let mut file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .unwrap();
        let mut header = [0u8; 512];
        file.read_exact(&mut header).unwrap();
        header[0] ^= 1;
        file.seek(SeekFrom::Start(0)).unwrap();
        file.write_all(&header).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();
        let archive_bytes = file.metadata().unwrap().len();
        assert!(
            preflight_tar(&mut file, archive_bytes)
                .unwrap_err()
                .contains("header checksum mismatch")
        );
        std::fs::remove_file(path).unwrap();
    }
}
