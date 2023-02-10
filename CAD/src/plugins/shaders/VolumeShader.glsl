precision highp float;
precision mediump sampler3D;

uniform vec3 u_size;
uniform int u_renderstyle;
uniform float u_renderthreshold;
uniform vec2 u_clim;
uniform vec2 u_clim_h;
uniform vec3 u_size_h;

uniform sampler3D u_data;
uniform sampler3D u_heat;
uniform sampler2D u_colormap;
uniform sampler2D u_colormap_h;
uniform float u_max;
uniform float u_min;
uniform float u_refl;
uniform float u_high;
uniform float u_low;

uniform bool u_show_heat;
uniform float u_heat_rate;

uniform float u_opacity;

varying vec3 v_position;
varying vec4 v_nearpos;
varying vec4 v_farpos;

// The maximum distance through our rendering volume is sqrt(3).
const int MAX_STEPS = 887;  // 887 for 512^3, 1774 for 1024^3
const int REFINEMENT_STEPS = 4;
const float relative_step_size = 1.0;
const vec4 ambient_color = vec4(0.2, 0.4, 0.2, 1.0);
const vec4 diffuse_color = vec4(0.8, 0.2, 0.2, 1.0);
const vec4 specular_color = vec4(1.0, 1.0, 1.0, 1.0);
const float shininess = 40.0;

vec4 cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);
vec4 cast_thres(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);
vec4 cast_thres2(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);
vec4 cast_iso(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray);
vec4 cast_heat(vec3 start_loc_h, vec3 step_h, int nsteps_h, vec3 view_ray);

float sample1(vec3 texcoords);
float sample2(vec3 texcoords);
vec4 apply_colormap(float val);
vec4 apply_colormap2(float val);
vec4 add_lighting(float val, vec3 loc, vec3 step, vec3 view_ray);

void main() {
  // Normalize clipping plane info
  vec3 farpos = v_farpos.xyz / v_farpos.w;
  vec3 nearpos = v_nearpos.xyz / v_nearpos.w;
  // Calculate unit vector pointing in the view direction through this fragment.
  vec3 view_ray = normalize(nearpos.xyz - farpos.xyz);
  // Compute the (negative) distance to the front surface or near clipping plane.
  // v_position is the back face of the cuboid, so the initial distance calculated in the dot
  // product below is the distance from near clip plane to the back of the cuboid

  float distance = dot(nearpos - v_position, view_ray);
  distance = max(distance, min((-0.5 - v_position.x) / view_ray.x,
                              (u_size.x - 0.5 - v_position.x) / view_ray.x));
  distance = max(distance, min((-0.5 - v_position.y) / view_ray.y,
                              (u_size.y - 0.5 - v_position.y) / view_ray.y));
  distance = max(distance, min((-0.5 - v_position.z) / view_ray.z,
                              (u_size.z - 0.5 - v_position.z) / view_ray.z));
  // Now we have the starting position on the front surface
  vec3 front = v_position + view_ray * distance;
  // Decide how many steps to take
  int nsteps = int(-distance / relative_step_size + 0.5);
  if ( nsteps < 1 )
    discard;
  // Get starting location and step vector in texture coordinates
  vec3 step = ((v_position - front) / u_size) / float(nsteps);
  vec3 start_loc = front / u_size;
  // For testing: show the number of steps. This helps to establish
  // whether the rays are correctly oriented
  // gl_FragColor = vec4(0.0, float(nsteps) / 1.0 / u_size.x, 1.0, 1.0);
  // return;
  vec4 model_color = vec4(0.0);
  if (u_renderstyle == 0)
    model_color = cast_thres(start_loc, step, nsteps, view_ray);
  else if (u_renderstyle == 1)
    model_color = cast_thres2(start_loc, step, nsteps, view_ray);
  else if (u_renderstyle == 2)
    model_color = cast_mip(start_loc, step, nsteps, view_ray);
  // else if (u_renderstyle == 3)
  //   model_color = cast_iso(start_loc, step, nsteps, view_ray);

  float distance_h = dot(nearpos - v_position, view_ray);
  distance_h = max(distance_h, min((-0.5 - v_position.x) / view_ray.x,
                              (u_size_h.x - 0.5 - v_position.x) / view_ray.x));
  distance_h = max(distance_h, min((-0.5 - v_position.y) / view_ray.y,
                              (u_size_h.y - 0.5 - v_position.y) / view_ray.y));
  distance_h = max(distance_h, min((-0.5 - v_position.z) / view_ray.z,
                              (u_size_h.z - 0.5 - v_position.z) / view_ray.z));
  int nsteps_h = int(-distance_h / relative_step_size + 0.5);
  vec4 heat_color = vec4(0.0);
  if (nsteps_h >= 1) {
    vec3 step_h = ((v_position - front) / u_size_h) / float(nsteps_h);
    vec3 start_loc_h = front / u_size_h;
    heat_color = cast_heat(start_loc_h, step, nsteps_h, view_ray);
  }

  if (u_show_heat) {
    // gl_FragColor = vec4(1.0);
    // gl_FragColor = heat_color;
    gl_FragColor = u_heat_rate * heat_color + (1.0 - u_heat_rate) * model_color;
  } else {
    gl_FragColor = model_color;
  }

  if (gl_FragColor.a < 0.05)
    discard;
}

float sample1(vec3 texcoords) {
    /* Sample float value from a 3D texture. Assumes intensity data. */
  float res = texture(u_data, texcoords.xyz).r;
  return res;
}

float sample2(vec3 texcoords) {
  float res = texture(u_heat, texcoords.xyz).r;
  return res;
}

vec4 apply_colormap(float val) {
  val = (val - u_clim[0]) / (u_clim[1] - u_clim[0]);
  return texture2D(u_colormap, vec2(val, 0.8));
}

vec4 apply_colormap2(float val) {
  val = (val - u_clim_h[0]) / (u_clim_h[1] - u_clim_h[0]);
  return texture2D(u_colormap_h, vec2(val, 0.8));
}


vec4 cast_heat(vec3 start_loc_h, vec3 step_h, int nsteps_h, vec3 view_ray) {
  float max_val = -1e6;
  int max_i = 0;
  vec3 loc = start_loc_h;

  // Enter the raycasting loop. In WebGL 1 the loop index cannot be compared with
  // non-constant expression. So we use a hard-coded max, and an additional condition
  // inside the loop.
  for (int iter = 0; iter < MAX_STEPS; iter++) {
    if (iter >= nsteps_h)
      break;
    // Sample from the 3D texture
    float val = sample2(loc);
    // Apply MIP operation
    if (val > max_val) {
      max_val = val;
      max_i = iter;
    }
    // Advance location deeper into the volume
    loc += step_h;
  }

  // Refine location, gives crispier images
  vec3 iloc = start_loc_h + step_h * (float(max_i) - 0.5);
  vec3 istep = step_h / float(REFINEMENT_STEPS);
  for (int i = 0; i < REFINEMENT_STEPS; i++) {
    max_val = max(max_val, sample2(iloc));
    iloc += istep;
  }

  // Resolve final color
  return apply_colormap2(max_val);
}

vec4 cast_mip(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {

  float max_val = -1e6;
  int max_i = 0;
  vec3 loc = start_loc;
  bool iso_found = false;
  float low_threshold = u_renderthreshold - 0.02 * (u_clim[1] - u_clim[0]);
  vec4 iso_color = vec4(0.0);
  // Enter the raycasting loop. In WebGL 1 the loop index cannot be compared with
  // non-constant expression. So we use a hard-coded max, and an additional condition
  // inside the loop.
  for (int iter=0; iter<MAX_STEPS; iter++) {
    if (iter >= nsteps)
      break;
    // Sample from the 3D texture
    float val = sample1(loc);
    if (!iso_found) {
      if (val > low_threshold) {
        iso_color = add_lighting(0.5, loc, 1.5 / u_size, view_ray);
        iso_found = true;
      }
    }
    // Apply MIP operation
    if (val > max_val) {
      max_val = val;
      max_i = iter;
    }
    // Advance location deeper into the volume
    loc += step;
  }

  // Refine location, gives crispier images
  vec3 iloc = start_loc + step * (float(max_i) - 0.5);
  vec3 istep = step / float(REFINEMENT_STEPS);
  for (int i=0; i<REFINEMENT_STEPS; i++) {
    max_val = max(max_val, sample1(iloc));
    iloc += istep;
  }

  // Resolve final color
  return apply_colormap(max_val) * (1.0 - u_opacity) + iso_color * u_opacity;
}


vec4 cast_thres(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {
  float rate = 0.2;
  float ray = 0.0;

  float val3 = sample1(start_loc);
  float val0 = val3;
  float val1 = val3;
  float val2 = val3;
  vec3 loc = start_loc + step;
  float val4 = sample1(loc);

  bool light_on = false;

  bool iso_found = false;
  float low_threshold = u_renderthreshold - 0.02 * (u_clim[1] - u_clim[0]);
  vec4 iso_color = vec4(0.0);

  int state = 0;
  int old_state = 0;
  float high = u_high;
  float low = u_low;
  // 0 - inside no objects
  // 1 - inside low-density objects
  // 2 - inside norm-density objects
  // 3 - inside high-density objects

  float current_total = 0.0;
  int current_box_size = 0;

  int valid_count = 0;
  int m_count = 0;
  float m_dense = 0.0;

  float multiplier = 1.0;

  for (int i = 0; i < MAX_STEPS; ++i) {
    if (i > nsteps) break;
    val0 = val1;
    val1 = val2;
    val2 = val3;
    val3 = val4;
    if (!iso_found) {
      if (val3 > low_threshold) {
        iso_color = add_lighting(0.5, loc, 1.5 / u_size, view_ray);
        iso_found = true;
      }
    }
    if (i >= nsteps - 1) {
      val4 = val3;
    } else {
      loc += step;
      val4 = sample1(loc);
    }
    float val = (val0 + 4.0 * val1 + 7.0 * val2 + 4.0 * val3 + val4) / 17.0;
    old_state = state;
    switch (old_state) {
      case 0:
        if (val > high) {
          state = 3;
        } else if (val > low) {
          state = 2;
        } else if (val > 0.0001) {
          state = 1;
        }
        break;
      case 1:
        if (val > high) {
          state = 3;
        } else if (val > low) {
          state = 2;
        } else if (val > 0.0001) {
          current_box_size += 1;
          current_total += val;
        } else {
          state = 0;
        }
        break;
      case 2:
        if (val > high) {
          state = 3;
        } else if (val > low) {
          current_box_size += 1;
          current_total += val;
        } else if (val > 0.0001) {
          state = 1;
        } else {
          state = 0;
        }
        break;
      case 3:
        if (val > high) {
          current_box_size += 1;
          current_total += val;
        } else if (val > low) {
          state = 2;
        } else if (val > 0.0001) {
          state = 1;
        } else {
          state = 0;
        }
        break;
    }
    if (old_state != state) {
      if (old_state == 0) {
        if (!light_on) {
          light_on = true;
        }
      }
      float avg_dense = current_total / float(current_box_size);
      valid_count += current_box_size;
      if (old_state == 1) {
        if (current_box_size > 0) {
          multiplier = multiplier * (1.0 - rate * (u_low - avg_dense) / u_low);
        }
      } else if (old_state == 2) {
        m_dense += current_total;
        m_count += current_box_size;
      } else if (old_state == 3) {
        m_dense += current_total;
        if (current_box_size > 1) {
          multiplier = multiplier * (1.0 + rate * (avg_dense - u_high) / (1.0 - u_high));
        }
      }
      current_total = 0.0;
      current_box_size = 0;
    }
  }
  if (!light_on) {
    ray = 0.0;
  } else {
    ray = 0.5;
    float avg_dense = m_dense / float(valid_count);
    if (multiplier < 0.0001) {
      multiplier = 0.0001;
    }
    ray = ray - rate * (1.0 - avg_dense);
    ray = ray * multiplier;
    if (ray < 0.01) {
      ray = 0.01;
    } else if (ray > 0.99) {
      ray = 0.99;
    }
  }
  if (ray < 0.0) {
    ray = 0.0;
  } else if (ray > 1.0) {
    ray = 1.0;
  }
  return apply_colormap(ray) * (1.0 - u_opacity) + iso_color * u_opacity;
}

vec4 cast_thres2(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {
  float min_val = u_low;
  float max_val = u_high;
  int h_count = 0;
  int l_count = 0;
  int m_count = 0;
  vec3 loc = start_loc;
  bool iso_found = false;
  float low_threshold = u_renderthreshold - 0.02 * (u_clim[1] - u_clim[0]);
  vec4 iso_color = vec4(0.0);
  for (int i = 0; i < MAX_STEPS; ++i) {
    if (i > nsteps) break;
    float val = sample1(loc);
    if (!iso_found) {
      if (val > low_threshold) {
        iso_color = add_lighting(0.5, loc, 1.5 / u_size, view_ray);
        iso_found = true;
      }
    }
    if (val < 0.001) {
    } else if (val < min_val) {
      l_count += 1;
    } else if (val < max_val) {
      m_count += 1;
    } else {
      h_count += 1;
    }
    loc += step;
  }
  float total = (float(l_count + m_count + h_count));
  float final = (float(m_count) * 0.5 + float(h_count) * 1.0) / total;
  return apply_colormap(final) * (1.0 - u_opacity) + iso_color * u_opacity;
}


vec4 cast_iso(vec3 start_loc, vec3 step, int nsteps, vec3 view_ray) {

  vec4 target_color = vec4(0.0);  // init transparent
  vec4 color3 = vec4(0.0);  // final color
  vec3 dstep = 1.5 / u_size;  // step to sample derivative
  vec3 loc = start_loc;

  float low_threshold = u_renderthreshold - 0.02 * (u_clim[1] - u_clim[0]);

  // Enter the raycasting loop. In WebGL 1 the loop index cannot be compared with
  // non-constant expression. So we use a hard-coded max, and an additional condition
  // inside the loop.
  for (int iter=0; iter<MAX_STEPS; iter++) {
    if (iter >= nsteps)
      break;

  // Sample from the 3D texture
    float val = sample1(loc);

    if (val > low_threshold) {
      // Take the last interval in smaller steps
      vec3 iloc = loc - 0.5 * step;
      vec3 istep = step / float(REFINEMENT_STEPS);
      for (int i=0; i<REFINEMENT_STEPS; i++) {
        val = sample1(iloc);
        if (val > u_renderthreshold) {
          return add_lighting(val, iloc, dstep, view_ray);
        }
        iloc += istep;
      }
    }
    // Advance location deeper into the volume
    loc += step;
  }
  return vec4(0.0);
}

vec4 add_lighting(float val, vec3 loc, vec3 step, vec3 view_ray)
{
  // Calculate color by incorporating lighting

  // View direction
  vec3 V = normalize(view_ray);

  // calculate normal vector from gradient
  vec3 N;
  float val1, val2;
  val1 = sample1(loc + vec3(-step[0], 0.0, 0.0));
  val2 = sample1(loc + vec3(+step[0], 0.0, 0.0));
  N[0] = val1 - val2;
  val = max(max(val1, val2), val);
  val1 = sample1(loc + vec3(0.0, -step[1], 0.0));
  val2 = sample1(loc + vec3(0.0, +step[1], 0.0));
  N[1] = val1 - val2;
  val = max(max(val1, val2), val);
  val1 = sample1(loc + vec3(0.0, 0.0, -step[2]));
  val2 = sample1(loc + vec3(0.0, 0.0, +step[2]));
  N[2] = val1 - val2;
  val = max(max(val1, val2), val);

  float gm = length(N); // gradient magnitude
  N = normalize(N);

  // Flip normal so it points towards viewer
  float Nselect = float(dot(N, V) > 0.0);
  N = (2.0 * Nselect - 1.0) * N;  // ==  Nselect * N - (1.0-Nselect)*N;

  // Init colors
  vec4 ambient_color = vec4(0.0, 0.0, 0.0, 0.0);
  vec4 diffuse_color = vec4(0.0, 0.0, 0.0, 0.0);
  vec4 specular_color = vec4(0.0, 0.0, 0.0, 0.0);

  // note: could allow multiple lights
  for (int i=0; i<1; i++)
  {
    // Get light direction (make sure to prevent zero devision)
    vec3 L = normalize(view_ray);  //lightDirs[i];
    float lightEnabled = float( length(L) > 0.0 );
    L = normalize(L + (1.0 - lightEnabled));

    // Calculate lighting properties
    float lambertTerm = clamp(dot(N, L), 0.0, 1.0);
    vec3 H = normalize(L+V); // Halfway vector
    float specularTerm = pow(max(dot(H, N), 0.0), shininess);

    // Calculate mask
    float mask1 = lightEnabled;

    // Calculate colors
    ambient_color +=  mask1 * ambient_color;  // * gl_LightSource[i].ambient;
    diffuse_color +=  mask1 * lambertTerm;
    specular_color += mask1 * specularTerm * specular_color;
  }

// Calculate final color by componing different components
  vec4 final_color;
  vec4 color = apply_colormap(val);
  final_color = color * (ambient_color + diffuse_color) + specular_color;
  final_color.a = color.a;
  return final_color;
}