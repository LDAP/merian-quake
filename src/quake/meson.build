shaders = [
    'quake.comp',
    'clear.comp', 
]
included_files = files(
    'quake_water.glsl',
    'config.h',
    'layout.glsl',
    'envmap.glsl',
    'grid.h',
    'shading.glsl',
    # Not working currently..
    # merian_shaders + 'common/normal_encode.glsl',
    # merian_shaders + 'common/microfacet_sampling.glsl',
    # merian_shaders + 'common/von_mises_fisher.glsl',
)

foreach s : shaders
  src_files += custom_target(
    s + '.spv.h',
    input: s,
    output: s + '.spv.h',
    # -x save binary output as text-based 32-bit hexadecimal numbers, allows #include as header
    command: [glslc, '-V', '-I' + merian_shaders, '-I' + project_source_root + '/src/quake' ,'--target-env', 'vulkan1.3', '-x', '-o', '@OUTPUT@', '@INPUT@'],
    depend_files: included_files
  )
endforeach

src_files += files('quake_node.cpp')
