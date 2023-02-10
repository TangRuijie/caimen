import {
  Uniform
} from 'three/src/Three';

export const MyVolumeRenderShader: {
  uniforms: {
    u_size: Uniform;
    u_renderstyle: Uniform;
    u_renderthreshold: Uniform;
    u_clim: Uniform;
    u_clim_h: Uniform;
    u_data: Uniform;
    u_colormap: Uniform;
    u_colormap_h: Uniform;
    u_min: Uniform;
    u_max: Uniform;
    u_refl: Uniform;
    u_high: Uniform;
    u_low: Uniform;
    u_heat: Uniform;
    u_size_h: Uniform;
    u_show_heat: Uniform;
    u_heat_rate: Uniform;
    u_opacity: Uniform;
  };
  vertexShader: string;
  fragmentShader: string;
};
