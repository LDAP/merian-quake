shaders = [
    'quake.comp',
    'clear.comp',
    'volume.comp',
    'volume_forward_project.comp'
]

foreach s : shaders
  src_files += shader_generator.process(s, extra_args: ['--prefix', 'merian_quake'])
endforeach

src_files += files('render_markovchain.cpp')
