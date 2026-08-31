#include "radar_live.h"

#include <math.h>
#include <string.h>

#define RADAR_SYNC_BYTE 0x01U
#define RADAR_HEADER_SIZE 8U
#define RADAR_MAX_PAYLOAD 1024U
#define RADAR_FRAME_BUFFER_SIZE (RADAR_HEADER_SIZE + RADAR_MAX_PAYLOAD + 1U)
#define RADAR_TARGET_LIST_TYPE 0x0A04U
#define RADAR_SAMPLE_PERIOD_MS 50U
#define RADAR_MAX_SAMPLE_GAP_MS 250U
#define RADAR_INFERENCE_STRIDE 10U
#define RAD_TO_DEG 57.29577951308232f
#define TRAINING_RANGE_NEAR_M 0.5f
#define TRAINING_RANGE_FAR_M 6.0f
/* Training CSV stores the LD6002B sensor coordinates without a Z-axis flip. */
#define RADAR_MODEL_Z_SIGN (1.0f)

static const float feature_mean[AI_MODEL_FEATURE_COUNT] = {
    0.6251651048660278f,
    2.3458504676818848f,
    -0.788230836391449f,
    0.038907285779714584f,
    2.6832118034362793f,
    14.188167572021484f,
    -19.048274993896484f,
};

static const float feature_std[AI_MODEL_FEATURE_COUNT] = {
    0.7486507296562195f,
    0.9851225018501282f,
    0.34003353118896484f,
    2.0538766384124756f,
    0.983132004737854f,
    17.69748306274414f,
    9.395633697509766f,
};

static uint8_t frame_buffer[RADAR_FRAME_BUFFER_SIZE];
static uint16_t frame_length;
static uint16_t expected_frame_length;
static float model_window[AI_MODEL_FRAME_COUNT * AI_MODEL_FEATURE_COUNT];
static float x_window[AI_MODEL_FRAME_COUNT];
static float y_window[AI_MODEL_FRAME_COUNT];
static float range_window[AI_MODEL_FRAME_COUNT];
static float z_window[AI_MODEL_FRAME_COUNT];
static uint32_t window_count;
static uint32_t samples_since_inference;
static uint32_t last_sample_ms;
static uint8_t have_sample_time;
static uint32_t next_sample_ms;
static uint8_t scheduler_started;
static uint32_t valid_frame_count;
static uint32_t invalid_frame_count;
static float latest_x;
static float latest_y;
static float latest_sensor_z;
static float latest_z;
static float latest_range;
static int32_t latest_dop_idx;
static int32_t latest_cluster_id;
static uint32_t latest_target_count;
static uint8_t latest_target_valid;
static uint8_t target_lost_pending;
static uint32_t latest_target_update_ms;
static uint32_t latest_target_generation;
static uint32_t last_sampled_generation;
static uint32_t last_sample_interval_ms;
static uint32_t gap_reset_count;
static uint32_t scheduler_late_count;
static uint32_t held_sample_count;
static uint32_t target_update_count;
static uint32_t rejected_target_count;
static uint32_t target_switch_count;
static uint32_t target_lost_count;

static float Median(float *values, uint32_t count)
{
  uint32_t i;
  uint32_t j;

  for (i = 1U; i < count; ++i)
  {
    float value = values[i];
    j = i;
    while ((j > 0U) && (values[j - 1U] > value))
    {
      values[j] = values[j - 1U];
      --j;
    }
    values[j] = value;
  }

  if ((count & 1U) != 0U)
  {
    return values[count / 2U];
  }
  return 0.5f * (values[(count / 2U) - 1U] + values[count / 2U]);
}

static uint8_t TinyFrameChecksum(const uint8_t *data, uint16_t length)
{
  uint8_t value = 0U;
  uint16_t index;

  for (index = 0U; index < length; ++index)
  {
    value ^= data[index];
  }
  return (uint8_t)(~value);
}

static uint16_t ReadBigEndianU16(const uint8_t *data)
{
  return (uint16_t)(((uint16_t)data[0] << 8U) | data[1]);
}

static int32_t ReadLittleEndianI32(const uint8_t *data)
{
  uint32_t value = ((uint32_t)data[0]) |
                   ((uint32_t)data[1] << 8U) |
                   ((uint32_t)data[2] << 16U) |
                   ((uint32_t)data[3] << 24U);
  return (int32_t)value;
}

static float ReadLittleEndianFloat(const uint8_t *data)
{
  uint32_t bits = ((uint32_t)data[0]) |
                  ((uint32_t)data[1] << 8U) |
                  ((uint32_t)data[2] << 16U) |
                  ((uint32_t)data[3] << 24U);
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
}

static RadarLiveEvent AddLatestTargetSample(uint32_t now_ms)
{
  float raw[AI_MODEL_FEATURE_COUNT];
  float horizontal;
  float range;
  float x = latest_x;
  float y = latest_y;
  float z = latest_z;
  int32_t dop_idx = latest_dop_idx;
  uint32_t feature;
  float *destination;

  horizontal = sqrtf((x * x) + (y * y));
  range = sqrtf((horizontal * horizontal) + (z * z));
  if (have_sample_time != 0U)
  {
    uint32_t elapsed = now_ms - last_sample_ms;
    last_sample_interval_ms = elapsed;
    if (elapsed > RADAR_MAX_SAMPLE_GAP_MS)
    {
      window_count = 0U;
      samples_since_inference = 0U;
      ++gap_reset_count;
    }
  }
  have_sample_time = 1U;
  last_sample_ms = now_ms;
  if (latest_target_generation == last_sampled_generation)
  {
    ++held_sample_count;
  }
  last_sampled_generation = latest_target_generation;

  raw[0] = x;
  raw[1] = y;
  raw[2] = z;
  raw[3] = (float)dop_idx;
  raw[4] = range;
  raw[5] = atan2f(x, y) * RAD_TO_DEG;
  raw[6] = atan2f(z, horizontal) * RAD_TO_DEG;

  if (window_count >= AI_MODEL_FRAME_COUNT)
  {
    memmove(model_window,
            model_window + AI_MODEL_FEATURE_COUNT,
            sizeof(float) * (AI_MODEL_FRAME_COUNT - 1U) * AI_MODEL_FEATURE_COUNT);
    memmove(x_window, x_window + 1U,
            sizeof(float) * (AI_MODEL_FRAME_COUNT - 1U));
    memmove(y_window, y_window + 1U,
            sizeof(float) * (AI_MODEL_FRAME_COUNT - 1U));
    memmove(range_window, range_window + 1U,
            sizeof(float) * (AI_MODEL_FRAME_COUNT - 1U));
    memmove(z_window, z_window + 1U,
            sizeof(float) * (AI_MODEL_FRAME_COUNT - 1U));
    destination = model_window +
                  ((AI_MODEL_FRAME_COUNT - 1U) * AI_MODEL_FEATURE_COUNT);
    x_window[AI_MODEL_FRAME_COUNT - 1U] = x;
    y_window[AI_MODEL_FRAME_COUNT - 1U] = y;
    range_window[AI_MODEL_FRAME_COUNT - 1U] = range;
    z_window[AI_MODEL_FRAME_COUNT - 1U] = z;
  }
  else
  {
    destination = model_window + (window_count * AI_MODEL_FEATURE_COUNT);
    x_window[window_count] = x;
    y_window[window_count] = y;
    range_window[window_count] = range;
    z_window[window_count] = z;
    ++window_count;
  }

  for (feature = 0U; feature < AI_MODEL_FEATURE_COUNT; ++feature)
  {
    destination[feature] = (raw[feature] - feature_mean[feature]) /
                           feature_std[feature];
  }

  ++samples_since_inference;
  if ((window_count == AI_MODEL_FRAME_COUNT) &&
      ((samples_since_inference >= RADAR_INFERENCE_STRIDE) ||
       (samples_since_inference == AI_MODEL_FRAME_COUNT)))
  {
    samples_since_inference = 0U;
    return RADAR_LIVE_INFERENCE_READY;
  }
  return RADAR_LIVE_SAMPLE_ADDED;
}

static RadarLiveEvent DecodeCompleteFrame(uint32_t now_ms)
{
  uint16_t payload_length = ReadBigEndianU16(&frame_buffer[3]);
  uint16_t message_type = ReadBigEndianU16(&frame_buffer[5]);
  const uint8_t *payload = &frame_buffer[RADAR_HEADER_SIZE];
  int32_t target_count;
  float target_x[10];
  float target_y[10];
  float target_sensor_z[10];
  float target_z[10];
  float target_range[10];
  int32_t target_dop_idx[10];
  int32_t target_cluster_id[10];
  int32_t index;
  int32_t first_valid_index = -1;
  int32_t continuity_index = -1;
  int32_t selected_index;
  float smallest_continuity_jump = 1000000.0f;

  if (TinyFrameChecksum(frame_buffer, 7U) != frame_buffer[7] ||
      TinyFrameChecksum(payload, payload_length) !=
          frame_buffer[RADAR_HEADER_SIZE + payload_length])
  {
    ++invalid_frame_count;
    return RADAR_LIVE_NO_EVENT;
  }

  ++valid_frame_count;
  if ((message_type != RADAR_TARGET_LIST_TYPE) || (payload_length < 4U))
  {
    return RADAR_LIVE_NO_EVENT;
  }

  target_count = ReadLittleEndianI32(payload);
  if ((target_count < 0) || (target_count > 10) ||
      (payload_length != (uint16_t)(4 + (target_count * 20))))
  {
    ++invalid_frame_count;
    return RADAR_LIVE_NO_EVENT;
  }

  latest_target_count = (uint32_t)target_count;
  if (target_count == 0)
  {
    if ((latest_target_valid != 0U) || (window_count != 0U))
    {
      target_lost_pending = 1U;
      ++target_lost_count;
    }
    latest_target_valid = 0U;
    scheduler_started = 0U;
    have_sample_time = 0U;
    window_count = 0U;
    samples_since_inference = 0U;
    return RADAR_LIVE_NO_EVENT;
  }

  for (index = 0; index < target_count; ++index)
  {
    const uint8_t *record = payload + 4U + ((uint32_t)index * 20U);
    float horizontal;
    target_x[index] = ReadLittleEndianFloat(record);
    target_y[index] = ReadLittleEndianFloat(record + 4U);
    target_sensor_z[index] = ReadLittleEndianFloat(record + 8U);
    target_z[index] = target_sensor_z[index] * RADAR_MODEL_Z_SIGN;
    target_dop_idx[index] = ReadLittleEndianI32(record + 12U);
    target_cluster_id[index] = ReadLittleEndianI32(record + 16U);
    horizontal = sqrtf((target_x[index] * target_x[index]) +
                       (target_y[index] * target_y[index]));
    target_range[index] = sqrtf((horizontal * horizontal) +
                                (target_z[index] * target_z[index]));
    if (isfinite(target_x[index]) && isfinite(target_y[index]) &&
        isfinite(target_z[index]) && isfinite(target_range[index]) &&
        (target_range[index] >= 0.3f) && (target_range[index] <= 8.0f) &&
        (target_z[index] >= -3.0f) && (target_z[index] <= 2.0f))
    {
      float dx;
      float dy;
      float dz;
      float jump;
      if (first_valid_index < 0)
        first_valid_index = index;
      if ((latest_target_valid != 0U) &&
          (target_cluster_id[index] == latest_cluster_id))
      {
        dx = target_x[index] - latest_x;
        dy = target_y[index] - latest_y;
        dz = target_z[index] - latest_z;
        jump = sqrtf((dx * dx) + (dy * dy) + (dz * dz));
        if (jump < smallest_continuity_jump)
        {
          smallest_continuity_jump = jump;
          continuity_index = index;
        }
      }
    }
  }

  selected_index = (continuity_index >= 0) ? continuity_index : first_valid_index;
  if (selected_index < 0)
  {
    ++rejected_target_count;
    return RADAR_LIVE_NO_EVENT;
  }
  if ((latest_target_valid != 0U) &&
      (target_cluster_id[selected_index] != latest_cluster_id))
  {
    ++target_switch_count;
  }
  latest_x = target_x[selected_index];
  latest_y = target_y[selected_index];
  latest_sensor_z = target_sensor_z[selected_index];
  latest_z = target_z[selected_index];
  latest_range = target_range[selected_index];
  latest_dop_idx = target_dop_idx[selected_index];
  latest_cluster_id = target_cluster_id[selected_index];
  latest_target_update_ms = now_ms;
  latest_target_valid = 1U;
  ++latest_target_generation;
  ++target_update_count;
  return RADAR_LIVE_NO_EVENT;
}

void RadarLive_Init(void)
{
  frame_length = 0U;
  expected_frame_length = 0U;
  window_count = 0U;
  samples_since_inference = 0U;
  have_sample_time = 0U;
  next_sample_ms = 0U;
  scheduler_started = 0U;
  valid_frame_count = 0U;
  invalid_frame_count = 0U;
  latest_x = 0.0f;
  latest_y = 0.0f;
  latest_sensor_z = 0.0f;
  latest_z = 0.0f;
  latest_range = 0.0f;
  latest_dop_idx = 0;
  latest_cluster_id = -1;
  latest_target_count = 0U;
  latest_target_valid = 0U;
  target_lost_pending = 0U;
  latest_target_update_ms = 0U;
  latest_target_generation = 0U;
  last_sampled_generation = 0U;
  last_sample_interval_ms = 0U;
  gap_reset_count = 0U;
  scheduler_late_count = 0U;
  held_sample_count = 0U;
  target_update_count = 0U;
  rejected_target_count = 0U;
  target_switch_count = 0U;
  target_lost_count = 0U;
  memset(model_window, 0, sizeof(model_window));
  memset(x_window, 0, sizeof(x_window));
  memset(y_window, 0, sizeof(y_window));
  memset(range_window, 0, sizeof(range_window));
  memset(z_window, 0, sizeof(z_window));
}

RadarLiveEvent RadarLive_ProcessByte(uint8_t byte, uint32_t now_ms)
{
  RadarLiveEvent event;

  if (frame_length == 0U)
  {
    if (byte != RADAR_SYNC_BYTE)
    {
      return RADAR_LIVE_NO_EVENT;
    }
    frame_buffer[frame_length++] = byte;
    return RADAR_LIVE_NO_EVENT;
  }

  if (frame_length >= RADAR_FRAME_BUFFER_SIZE)
  {
    frame_length = 0U;
    expected_frame_length = 0U;
    ++invalid_frame_count;
    return RADAR_LIVE_NO_EVENT;
  }
  frame_buffer[frame_length++] = byte;

  if (frame_length == RADAR_HEADER_SIZE)
  {
    uint16_t payload_length = ReadBigEndianU16(&frame_buffer[3]);
    if ((payload_length > RADAR_MAX_PAYLOAD) ||
        (TinyFrameChecksum(frame_buffer, 7U) != frame_buffer[7]))
    {
      frame_length = 0U;
      expected_frame_length = 0U;
      ++invalid_frame_count;
      return RADAR_LIVE_NO_EVENT;
    }
    expected_frame_length = (uint16_t)(RADAR_HEADER_SIZE + payload_length + 1U);
  }

  if ((expected_frame_length != 0U) && (frame_length == expected_frame_length))
  {
    event = DecodeCompleteFrame(now_ms);
    frame_length = 0U;
    expected_frame_length = 0U;
    return event;
  }
  return RADAR_LIVE_NO_EVENT;
}

RadarLiveEvent RadarLive_Tick(uint32_t now_ms)
{
  RadarLiveEvent event;

  if (target_lost_pending != 0U)
  {
    target_lost_pending = 0U;
    return RADAR_LIVE_TARGET_LOST;
  }

  if (latest_target_valid == 0U)
  {
    return RADAR_LIVE_NO_EVENT;
  }

  if (scheduler_started == 0U)
  {
    scheduler_started = 1U;
    next_sample_ms = now_ms;
  }

  if ((int32_t)(now_ms - next_sample_ms) < 0)
  {
    return RADAR_LIVE_NO_EVENT;
  }

  event = AddLatestTargetSample(now_ms);
  next_sample_ms += RADAR_SAMPLE_PERIOD_MS;
  if ((int32_t)(now_ms - next_sample_ms) >= 0)
  {
    next_sample_ms = now_ms + RADAR_SAMPLE_PERIOD_MS;
    ++scheduler_late_count;
  }
  return event;
}

const float *RadarLive_GetModelInput(void)
{
  return model_window;
}

uint32_t RadarLive_GetWindowCount(void)
{
  return window_count;
}

uint32_t RadarLive_GetValidFrameCount(void)
{
  return valid_frame_count;
}

uint32_t RadarLive_GetInvalidFrameCount(void)
{
  return invalid_frame_count;
}

void RadarLive_GetDiagnostics(RadarLiveDiagnostics *diagnostics)
{
  uint32_t index;
  uint32_t value_count;
  float normalized_abs_max = 0.0f;
  float pre_z[20];
  float post_z[10];
  float post_motion[9];

  if (diagnostics == NULL)
  {
    return;
  }

  memset(diagnostics, 0, sizeof(*diagnostics));
  diagnostics->latest_x = latest_x;
  diagnostics->latest_y = latest_y;
  diagnostics->latest_sensor_z = latest_sensor_z;
  diagnostics->latest_z = latest_z;
  diagnostics->latest_range = latest_range;
  diagnostics->latest_azimuth_deg = atan2f(latest_x, latest_y) * RAD_TO_DEG;
  diagnostics->latest_dop_idx = latest_dop_idx;
  diagnostics->latest_cluster_id = latest_cluster_id;
  diagnostics->latest_target_count = latest_target_count;
  diagnostics->last_sample_interval_ms = last_sample_interval_ms;
  diagnostics->latest_target_age_ms =
      (latest_target_valid != 0U) ? (last_sample_ms - latest_target_update_ms) : 0U;
  diagnostics->gap_reset_count = gap_reset_count;
  diagnostics->scheduler_late_count = scheduler_late_count;
  diagnostics->held_sample_count = held_sample_count;
  diagnostics->target_update_count = target_update_count;
  diagnostics->rejected_target_count = rejected_target_count;
  diagnostics->target_switch_count = target_switch_count;
  diagnostics->target_lost_count = target_lost_count;
  if (latest_range < TRAINING_RANGE_NEAR_M)
  {
    diagnostics->range_quality = RADAR_RANGE_TOO_CLOSE;
  }
  else if (latest_range > TRAINING_RANGE_FAR_M)
  {
    diagnostics->range_quality = RADAR_RANGE_TOO_FAR;
  }
  else
  {
    diagnostics->range_quality = RADAR_RANGE_IN_TRAINING_SUPPORT;
  }

  if (window_count == 0U)
  {
    return;
  }

  diagnostics->window_range_min = range_window[0];
  diagnostics->window_range_max = range_window[0];
  diagnostics->window_z_min = z_window[0];
  diagnostics->window_z_max = z_window[0];
  for (index = 1U; index < window_count; ++index)
  {
    if (range_window[index] < diagnostics->window_range_min)
      diagnostics->window_range_min = range_window[index];
    if (range_window[index] > diagnostics->window_range_max)
      diagnostics->window_range_max = range_window[index];
    if (z_window[index] < diagnostics->window_z_min)
      diagnostics->window_z_min = z_window[index];
    if (z_window[index] > diagnostics->window_z_max)
      diagnostics->window_z_max = z_window[index];
  }
  diagnostics->window_range_delta =
      range_window[window_count - 1U] - range_window[0];
  diagnostics->window_z_delta = z_window[window_count - 1U] - z_window[0];

  if (window_count == AI_MODEL_FRAME_COUNT)
  {
    float downward_sum = 0.0f;
    float vertical_motion_sum = 0.0f;
    for (index = 0U; index < 20U; ++index)
    {
      pre_z[index] = z_window[index];
    }
    for (index = 0U; index < 10U; ++index)
    {
      post_z[index] = z_window[AI_MODEL_FRAME_COUNT - 10U + index];
    }
    for (index = 0U; index < 9U; ++index)
    {
      uint32_t first = AI_MODEL_FRAME_COUNT - 10U + index;
      float dx = x_window[first + 1U] - x_window[first];
      float dy = y_window[first + 1U] - y_window[first];
      float dz = z_window[first + 1U] - z_window[first];
      post_motion[index] = sqrtf((dx * dx) + (dy * dy) + (dz * dz));
    }
    diagnostics->pre_event_z_median = Median(pre_z, 20U);
    diagnostics->post_event_z_median = Median(post_z, 10U);
    diagnostics->height_drop = diagnostics->pre_event_z_median -
                               diagnostics->post_event_z_median;
    for (index = 1U; index < AI_MODEL_FRAME_COUNT; ++index)
    {
      float vertical_delta = z_window[index] - z_window[index - 1U];
      vertical_motion_sum += fabsf(vertical_delta);
      if (vertical_delta < 0.0f)
      {
        downward_sum -= vertical_delta;
      }
    }
    diagnostics->downward_fraction =
        (vertical_motion_sum > 0.000001f) ?
        (downward_sum / vertical_motion_sum) : 0.0f;
    diagnostics->post_motion_median = Median(post_motion, 9U);
  }

  value_count = window_count * AI_MODEL_FEATURE_COUNT;
  for (index = 0U; index < value_count; ++index)
  {
    float absolute_value = fabsf(model_window[index]);
    if (absolute_value > normalized_abs_max)
      normalized_abs_max = absolute_value;
  }
  diagnostics->normalized_abs_max = normalized_abs_max;
}
