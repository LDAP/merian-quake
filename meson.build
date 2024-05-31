project(
    'merian-quake',
    ['cpp', 'c'],
    version : '0.0.1',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
        'buildtype=release',
    ]
)

add_project_arguments('-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1', language : 'cpp')
add_project_arguments('-DMERIAN_QUAKE_RESOURCES="@0@"'.format(join_paths(get_option('prefix'), 'res')), language : 'cpp')

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

if get_option('buildtype') == 'debug'
  add_project_arguments('-fsanitize=address,undefined', language : 'cpp')
  add_project_link_arguments('-fsanitize=address,undefined', language : 'cpp')
endif

if get_option('performance_profiling')
  add_project_arguments('-DMERIAN_PROFILER_ENABLE', language : 'cpp')
endif

# Dependencies
merian = dependency('merian', fallback: ['merian', 'merian_dep'])
quake = dependency('quakespasm', fallback: ['quakespasm', 'quake_dep'])

# Shaders
glslc = find_program('glslangValidator')

src_files = []
inc_dirs = [
    include_directories('./src')
]

project_source_root = meson.project_source_root()
merian_shaders = project_source_root + '/subprojects/merian/src/merian-nodes'
merian_include = project_source_root + '/subprojects/merian/src'

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

# exe_headless = executable(
#     'merian-quake-headless',
#     [src_files, 'src/merian-quake-headless.cpp'],
#     dependencies: [
#         merian,
#         quake,
#     ],
#     include_directories: inc_dirs,
#     install : true,
# )

install_subdir('res', install_dir : '.')
