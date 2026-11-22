__device__ inline float evaluate_opacity_factor(const float dx,
                                                const float dy,
                                                const float4 co) {
  return 0.5f * (co.x * dx * dx + co.z * dy * dy) + co.y * dx * dy;
}

template <uint32_t PATCH_WIDTH, uint32_t PATCH_HEIGHT>
__device__ inline float max_contrib_power_rect_gaussian_float(
    const float4 co,
    const float2 mean,
    const glm::vec2 rect_min,
    const glm::vec2 rect_max,
    glm::vec2& max_pos) {
  const float x_min_diff = rect_min.x - mean.x;
  const float x_left = x_min_diff > 0.0f;
  // const float x_left = mean.x < rect_min.x;
  const float not_in_x_range = x_left + (mean.x > rect_max.x);

  const float y_min_diff = rect_min.y - mean.y;
  const float y_above = y_min_diff > 0.0f;
  // const float y_above = mean.y < rect_min.y;
  const float not_in_y_range = y_above + (mean.y > rect_max.y);

  max_pos = {mean.x, mean.y};
  float max_contrib_power = 0.0f;

  if ((not_in_y_range + not_in_x_range) > 0.0f) {
    const float px = x_left * rect_min.x + (1.0f - x_left) * rect_max.x;
    const float py = y_above * rect_min.y + (1.0f - y_above) * rect_max.y;

    const float dx = copysign(float(PATCH_WIDTH), x_min_diff);
    const float dy = copysign(float(PATCH_HEIGHT), y_min_diff);

    const float diffx = mean.x - px;
    const float diffy = mean.y - py;

    const float rcp_dxdxcox =
        __frcp_rn(PATCH_WIDTH * PATCH_WIDTH * co.x);  // = 1.0 / (dx*dx*co.x)
    const float rcp_dydycoz =
        __frcp_rn(PATCH_HEIGHT * PATCH_HEIGHT * co.z);  // = 1.0 / (dy*dy*co.z)

    const float tx =
        not_in_y_range *
        __saturatef((dx * co.x * diffx + dx * co.y * diffy) * rcp_dxdxcox);
    const float ty =
        not_in_x_range *
        __saturatef((dy * co.y * diffx + dy * co.z * diffy) * rcp_dydycoz);
    max_pos = {px + tx * dx, py + ty * dy};

    const float2 max_pos_diff = {mean.x - max_pos.x, mean.y - max_pos.y};
    max_contrib_power =
        evaluate_opacity_factor(max_pos_diff.x, max_pos_diff.y, co);
  }

  return max_contrib_power;
}


__global__ void emitSerial(int P,
                                  const float2* points_xy,
                                  const float4* __restrict__ conic_opacity,
                                  const float* depths,
                                  const uint32_t* offsets,
                                  uint64_t* gaussian_keys_unsorted,
                                  uint32_t* gaussian_values_unsorted,
                                  int* radii,
                                  dim3 grid,
                                  int2* rects) {
  auto idx = cg::this_grid().thread_rank();
  if (idx >= P) return;

  // Generate no key/value pair for invisible Gaussians
  if (radii[idx] > 0) {
    // Find this Gaussian's offset in buffer for writing keys/values.
    uint32_t off = (idx == 0) ? 0 : offsets[idx - 1];
    const uint32_t offset_to = offsets[idx];
    if (offset_to < off) return;
    uint2 rect_min, rect_max;

    if (rects == nullptr)
      getRect(points_xy[idx], radii[idx], rect_min, rect_max, grid);
    else
      getRect(points_xy[idx], rects[idx], rect_min, rect_max, grid);

    const float2 xy = points_xy[idx];
    const float4 co = conic_opacity[idx];
    const float opacity_threshold = 1.0f / 255.0f;
    const float opacity_factor_threshold = logf(co.w / opacity_threshold);

    // For each tile that the bounding rect overlaps, emit a
    // key/value pair. The key is |  tile ID  |      depth      |,
    // and the value is the ID of the Gaussian. Sorting the values
    // with this key yields Gaussian IDs in a list, such that they
    // are first sorted by tile and then by depth.
    for (int y = rect_min.y; y < rect_max.y; y++) {
      for (int x = rect_min.x; x < rect_max.x; x++) {
        if (off >= offset_to) break;
        const glm::vec2 tile_min(x * BLOCK_X, y * BLOCK_Y);
        const glm::vec2 tile_max((x + 1) * BLOCK_X - 1, (y + 1) * BLOCK_Y - 1);

        glm::vec2 max_pos;
        float max_opac_factor = 0.0f;
        max_opac_factor =
            max_contrib_power_rect_gaussian_float<BLOCK_X - 1, BLOCK_Y - 1>(
                co, xy, tile_min, tile_max, max_pos);

        uint64_t key = y * grid.x + x;
        key <<= 32;
        key |= *((uint32_t*)&depths[idx]);
        if (max_opac_factor <= opacity_factor_threshold) {
          gaussian_keys_unsorted[off] = key;
          gaussian_values_unsorted[off] = idx;
          off++;
        }
      }
    }

    for (; off < offset_to; ++off) {
      uint64_t key = (uint32_t)-1;
      key <<= 32;
      const float depth = FLT_MAX;
      key |= *((uint32_t*)&depth);
      gaussian_values_unsorted[off] = static_cast<uint32_t>(-1);
      gaussian_keys_unsorted[off] = key;
    }
  }
}

