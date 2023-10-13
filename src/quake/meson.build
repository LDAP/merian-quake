shaders = [
    'quake.comp',
    'clear.comp',
    'volume.comp',
    'volume_forward_project.comp'
]
included_files = files(
    'quake_water.glsl',
    'config.h',
    'layout.glsl',
    'grid.h',
    'shading.glsl',
    'mc.glsl',
    'mc_distance.glsl',
    'light_cache.glsl',
    'raytrace.glsl'
    # Not working currently..
    # merian_shaders + 'common/normal_encode.glsl',
    # merian_shaders + 'common/microfacet_sampling.glsl',
    # merian_shaders + 'common/von_mises_fisher.glsl',
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
