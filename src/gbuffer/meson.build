shaders = [
    'gbuffer.comp',
]

foreach s : shaders
  src_files += shader_generator.process(s)
endforeach

src_files += files('gbuffer.cpp')
