shaders = [
    'hud.comp',
]

foreach s : shaders
  src_files += shader_generator.process(s)
endforeach

src_files += files('hud.cpp')
