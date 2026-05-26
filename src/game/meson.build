src_files += files(
    'quake_extraction.cpp',
    'quake_node.cpp',
    'quake_scene.cpp',
    'quake_ui_node.cpp',
)

src_files += shader_generator.process('quake_draw.vert',
                                      extra_args: ['--prefix', 'merian'],
                                      preserve_path_from: meson.current_source_dir())
src_files += shader_generator.process('quake_draw.frag',
                                      extra_args: ['--prefix', 'merian'],
                                      preserve_path_from: meson.current_source_dir())
