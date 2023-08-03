# Merian-Quake RT

This is a ray traced engine for [quakespasm](https://github.com/sezero/quakespasm).

## Usage

```bash
merian-quake <quakespasm arguments>
```

You can specify the path to your quake folder using the `-basedir` argument if you do not want to use the default.


## Building

Supported OS:

- Windows 11
- Linux

Build dependencies:

- Vulkan SDK
- Meson
- A fairly recent C++ compiler

```bash
# Clone the repository with all submodules:
git clone --recursive https://github.com/LDAP/merian-quake-rt
cd merian-quake-rt

# Compile
meson setup build [--prefix=path/to/installdir]
# or debug
meson setup build [--prefix=path/to/installdir] --buildtype=debug 

meson compile -C build

# Install
meson install -C build
```


## Development

```bash
# Unix
# Just run
./build/merian-quake [-basedir /path/to/quakedir]

# Windows: (requires .dll paths set up correctly)
meson devenv -C build
./build/merian-quake [-basedir /path/to/quakedir]
```
