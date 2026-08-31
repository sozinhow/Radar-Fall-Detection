#ifndef RADAR_LIVE_H
#define RADAR_LIVE_H

#include <stdint.h>

#include "ai_model.h"

typedef enum
{
  RADAR_LIVE_NO_EVENT = 0,
  RADAR_LIVE_SAMPLE_ADDED = 1,
  RADAR_LIVE_INFERENCE_READY = 2,
  RADAR_LIVE_TARGET_LOST = 3
} RadarLiveEvent;

typedef enum
{
  RADAR_RANGE_IN_TRAINING_SUPPORT = 0,
  RADAR_RANGE_TOO_CLOSE = 1,
  RADAR_RANGE_TOO_FAR = 2
} RadarRangeQuality;

typedef struct
{
  float latest_x;
  float latest_y;
  float latest_sensor_z;
  float latest_z;
  float latest_range;
  float latest_azimuth_deg;
  int32_t latest_dop_idx;
  int32_t latest_cluster_id;
  uint32_t latest_target_count;
  float window_range_min;
  float window_range_max;
  float window_range_delta;
  float window_z_min;
  float window_z_max;
  float window_z_delta;
  float pre_event_z_median;
  float post_event_z_median;
  float height_drop;
  float downward_fraction;
  float post_motion_median;
  float normalized_abs_max;
  uint32_t last_sample_interval_ms;
  uint32_t latest_target_age_ms;
  uint32_t gap_reset_count;
  uint32_t scheduler_late_count;
  uint32_t held_sample_count;
  uint32_t target_update_count;
  uint32_t rejected_target_count;
  uint32_t target_switch_count;
  uint32_t target_lost_count;
  RadarRangeQuality range_quality;
} RadarLiveDiagnostics;

void RadarLive_Init(void);
RadarLiveEvent RadarLive_ProcessByte(uint8_t byte, uint32_t now_ms);
RadarLiveEvent RadarLive_Tick(uint32_t now_ms);
const float *RadarLive_GetModelInput(void);
uint32_t RadarLive_GetWindowCount(void);
uint32_t RadarLive_GetValidFrameCount(void);
uint32_t RadarLive_GetInvalidFrameCount(void);
void RadarLive_GetDiagnostics(RadarLiveDiagnostics *diagnostics);

#endif /* RADAR_LIVE_H */
