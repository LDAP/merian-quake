src_files += shader_generator.process(
  'restir_di_generate_samples_bsdf.comp',
  'restir_di_temporal_reuse.comp',
  'restir_di_spatial_reuse.comp',
  'restir_di_shade.comp',
  'restir_di_clear.comp',
  extra_args: ['--prefix', 'merian_quake'] + glslc_args,
  preserve_path_from: meson.source_root(),
)


src_files += files('renderer_restir.cpp')
