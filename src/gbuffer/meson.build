shaders = [
    'gbuffer.comp',
]

foreach s : shaders
  src_files += shader_generator.process(s, extra_args: ['--prefix', 'merian_quake'])
endforeach

src_files += files('gbuffer.cpp')
