project(
    'merian-quake',
    ['cpp', 'c'],
    version : '0.0.1',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
        'buildtype=release',
    ],
    meson_version: '>=1.2.0',
)

data_subdir = join_paths(get_option('datadir'), meson.project_name())
add_project_arguments('-DMERIAN_QUAKE_DATA_DIR="@0@"'.format(join_paths(join_paths(get_option('prefix'), data_subdir))), language : 'cpp')
add_project_arguments('-DMERIAN_QUAKE_PROJECT_NAME="@0@"'.format(meson.project_name()), language : 'cpp')

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

# Dependencies
merian_subp = subproject('merian')
merian = merian_subp.get_variable('merian_dep')

quake = dependency('quakespasm', fallback: ['quakespasm', 'quake_dep'], default_options: {'snd_extern': true})

# Shaders
glslc = find_program('glslangValidator')

src_files = []
inc_dirs = [
    include_directories('./src')
]

shader_generator = merian_subp.get_variable('shader_generator')

subdir('src')

exe = executable(
    'merian-quake',
    [src_files, 'src/merian-quake.cpp'],
    dependencies: [
        merian,
        quake,
    ],
    include_directories: inc_dirs,
    install : true,
)

install_subdir('res', install_dir: data_subdir, strip_directory: true)
