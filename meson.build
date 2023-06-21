project(
    'merian-quake',
    'cpp',
    version : '0.0.1',
    default_options : [
        'warning_level=3',
        'cpp_std=c++20',
        'b_ndebug=if-release',
    ]
)

# Debug configuration
if get_option('buildtype').startswith('debug')
  add_project_arguments('-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_DEBUG', language : 'cpp')
endif

# Dependencies
fmt = dependency('fmt', version : ['>=8.1.1'], fallback : ['fmt', 'fmt_dep'])
spdlog = dependency('spdlog', version : ['>=1.10.0'], fallback : ['spdlog', 'spdlog_dep'], default_options : ['external_fmt=enabled'])
vulkan = dependency('vulkan', version: ['>=1.3.0'])
glm = dependency('glm', fallback : ['glm', 'glm_dep'])
merian = dependency('merian', fallback: ['merian', 'merian_dep'])

# Shaders
glslc = find_program('glslangValidator')
shaders = files(
    'res/shaders/shader.comp.glsl',
)
shader_targets = []
foreach s : shaders
    shader_targets += custom_target(
        'shader @0@'.format(s),
        command : [glslc, '--target-env', 'vulkan1.3', '-Ires/shaders', '-V', '@INPUT@', '-o',  '@OUTPUT@'],
        input : s, 
        output : '@PLAINNAME@.spv',
        install_dir: 'assets/shaders',
        install: true
    )
endforeach

src_files = files(
    'src/main.cpp',
)
inc_dirs = [
]

exe = executable(
    'merian-quake',
    src_files,
    dependencies: [
        fmt,
        spdlog,
        vulkan,
        merian,
        glm,
    ],
    include_directories: inc_dirs,
    install : true
)
