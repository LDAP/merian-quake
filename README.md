# Merian-Quake

This is a raytraced version of quakespasm.

## Usage

Copy you maps into `res/quake` (the id1 folder), or specify a custom path.
Then compile and run:

```bash
meson setup build
# or debug
meson setup build --buildtype=debug

meson compile -C build
./build/merian-quake
```
