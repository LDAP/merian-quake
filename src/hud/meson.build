shaders = [
    'hud.comp',
]

foreach s : shaders
  src_files += shader_generator.process(s, extra_args: ['--prefix', 'merian_quake'] + glslc_args)
endforeach

src_files += files('hud.cpp')
