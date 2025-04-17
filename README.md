# Merian-Quake

A path-tracer for the original Quake game on top of [quakespasm](https://github.com/sezero/quakespasm) and [Merian](https://github.com/LDAP/merian).

<hr>

This project showcases a novel real-time path guiding algorithm. For a comprehensive description of the algorithm and the inner workings of this project, please refer to the [paper](https://www.lalber.org/alber2025MCPG) and the [project website](https://www.lalber.org/2025/04/markov-chain-path-guiding/):

> Lucas Alber, Johannes Hanika, and Carsten Dachsbacher. 2025. Real-Time Markov Chain Path Guiding for Global Illumination and Single Scattering. Proc. ACM Comput. Graph. Interact. Tech. 8, 1, Article 15 (May 2025), 18 pages. https://doi.org/10.1145/3728296

<hr>

<p align="middle">
  <img src="images/alk.png" width="400" />
  <img src="images/azad_2.png" width="400" /> 
  <img src="images/sepulcher.png" width="400" /> 
  <img src="images/tears.png" width="400" /> 
</p>

Maps in screenshots: start from [Alkaline](https://alkalinequake.wordpress.com/) and ad_azad, ad_sepulcher, ad_tears from [Arcane Dimensions](https://www.moddb.com/mods/arcane-dimensions). 

## Licensing

This project is licensed under GPLv2.
The code for Markov Chain Path Guiding (`/res/shader/render_mcpg`) is licensed under GPLv3.
The code for ReSTIR (`/res/shader/render_restir`) and Screen-Space Mixture Models (`/res/shader/render_ssmm`) is licensed under the BSD 3-Clause License.

Get in touch if you are interested in a different licensing model.

## Requirements

This project requires a GPU with ray tracing support (Vulkan Ray Query).

Tested: NVIDIA GeForce RTX 2080 / 3070 TI / 3080 TI / 4080 TI, AMD Radeon RX 6800 XT, AMD RX 7900 XTX and Intel Arc.

## Usage

This project builds on [quakespasm](https://github.com/sezero/quakespasm), follow the instructions there on how to install the necessary `.pak` files. 

```bash
merian-quake <quakespasm arguments>
```

You can specify the path to your quake folder using the `-basedir` argument if you do not want to use the default.


## Building and Installing

Supported OS:

- Windows 11
- Linux

Build dependencies:

- Vulkan SDK
- Meson
- A fairly recent C++ compiler (Visual Studio Build Tools on Windows)

```bash
# Clone the repository with all submodules:
git clone --recursive https://github.com/LDAP/merian-quake
cd merian-quake

# Configure and compile
meson setup build [--prefix=path/to/installdir]
# or debug
meson setup build [--prefix=path/to/installdir] --buildtype=debug
# on windows you might need to add --backend vs or set the compiler to cl (MSVC) using environment variables

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
./merian-quake [-basedir /path/to/quakedir]
```

Note, the quakedir is the folder that contains the `id1` subfolder.
