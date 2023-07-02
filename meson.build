project(
    'merian-quake',
    ['cpp', 'c'],
    version : '0.0.1',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
    ]
)

add_project_arguments('-DVULKAN_HPP_DISPATCH_LOADER_DYNAMIC=1', language : 'cpp')

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

if get_option('performance_profiling')
  add_project_arguments('-DMERIAN_PROFILER_ENABLE', language : 'cpp')
endif

# Dependencies
fmt = dependency('fmt', version : ['>=9.1.0'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.11.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan', version: ['>=1.3.0'])
glm = dependency('glm', fallback : ['glm', 'glm_dep'])
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

subdir('src')

exe = executable(
    'merian-quake',
    src_files,
    dependencies: [
        fmt,
        glm,
        merian,
        quake,
        spdlog,
        vulkan,
    ],
    include_directories: inc_dirs,
    install : true,
)
