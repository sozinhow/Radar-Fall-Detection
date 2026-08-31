#ifndef AI_MODEL_H
#define AI_MODEL_H

#include <stdint.h>

#define AI_MODEL_FRAME_COUNT 60U
#define AI_MODEL_FEATURE_COUNT 7U
#define AI_MODEL_OUTPUT_COUNT 2U

int32_t AI_Model_Init(void);
int32_t AI_Model_Run(const float input[AI_MODEL_FRAME_COUNT * AI_MODEL_FEATURE_COUNT],
                     float output[AI_MODEL_OUTPUT_COUNT]);
int32_t AI_Model_RunSelfTest(float output[AI_MODEL_OUTPUT_COUNT]);

#endif /* AI_MODEL_H */
