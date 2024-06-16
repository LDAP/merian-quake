shaders = [
    'draw.vert',
    'draw.frag',
]
included_files = files(
    # Not working currently..
)

foreach s : shaders
  glslc_args = ['-I' + merian_shaders, '-I' + project_source_root + '/src/quake', '-I' + merian_include,'--target-env', 'vulkan1.3']
  compile_script = [project_source_root + '/subprojects/merian/scripts/compile_shader.py']

  # Header
  src_files += custom_target(
    s + '.spv.h',
    output: s + '.spv.h',
    command: compile_script + ['--header', s, '@OUTPUT@']
  )

  # Implementation
  src_files += custom_target(
    s + '.spv.c',
    input: s,
    output: s + '.spv.c',
    command: compile_script + ['@INPUT@', '@OUTPUT@'] + glslc_args,
    depend_files: included_files
  )
endforeach

src_files += files('quake_node.cpp')
