#include "ai_model.h"

#include "network.h"

#include <string.h>

STAI_NETWORK_CONTEXT_DECLARE(ai_network_context, STAI_NETWORK_CONTEXT_SIZE)

STAI_ALIGNED(32)
static uint8_t ai_activations[STAI_NETWORK_ACTIVATIONS_SIZE_BYTES];

static stai_ptr ai_activation_ptrs[STAI_NETWORK_ACTIVATIONS_NUM] = {
    ai_activations,
};
static stai_ptr ai_input_ptrs[STAI_NETWORK_IN_NUM];
static stai_ptr ai_output_ptrs[STAI_NETWORK_OUT_NUM];
static uint8_t ai_initialized;

int32_t AI_Model_Init(void)
{
  stai_return_code status;
  stai_size count;

  status = stai_network_init(ai_network_context);
  if (status != STAI_SUCCESS)
  {
    return (int32_t)status;
  }

  status = stai_network_set_activations(
      ai_network_context, ai_activation_ptrs, STAI_NETWORK_ACTIVATIONS_NUM);
  if (status != STAI_SUCCESS)
  {
    return (int32_t)status;
  }

  count = STAI_NETWORK_IN_NUM;
  status = stai_network_get_inputs(ai_network_context, ai_input_ptrs, &count);
  if ((status != STAI_SUCCESS) || (count != STAI_NETWORK_IN_NUM))
  {
    return (status != STAI_SUCCESS) ? (int32_t)status : -1001;
  }

  count = STAI_NETWORK_OUT_NUM;
  status = stai_network_get_outputs(ai_network_context, ai_output_ptrs, &count);
  if ((status != STAI_SUCCESS) || (count != STAI_NETWORK_OUT_NUM))
  {
    return (status != STAI_SUCCESS) ? (int32_t)status : -1002;
  }

  ai_initialized = 1U;
  return 0;
}

int32_t AI_Model_RunSelfTest(float output[AI_MODEL_OUTPUT_COUNT])
{
  static float test_input[AI_MODEL_FRAME_COUNT * AI_MODEL_FEATURE_COUNT];
  uint32_t frame;
  uint32_t feature;

  /* Deterministic 60x7 sample used only to verify the generated network. */
  for (frame = 0U; frame < AI_MODEL_FRAME_COUNT; ++frame)
  {
    for (feature = 0U; feature < AI_MODEL_FEATURE_COUNT; ++feature)
    {
      test_input[(frame * AI_MODEL_FEATURE_COUNT) + feature] =
          ((float)frame * 0.001f) + ((float)feature * 0.01f);
    }
  }

  return AI_Model_Run(test_input, output);
}

int32_t AI_Model_Run(
    const float input[AI_MODEL_FRAME_COUNT * AI_MODEL_FEATURE_COUNT],
    float output[AI_MODEL_OUTPUT_COUNT])
{
  float *network_input;
  float *network_output;
  stai_return_code status;

  if ((ai_initialized == 0U) || (input == NULL) || (output == NULL))
  {
    return -1003;
  }

  network_input = (float *)ai_input_ptrs[0];
  network_output = (float *)ai_output_ptrs[0];
  memcpy(network_input, input,
         sizeof(float) * AI_MODEL_FRAME_COUNT * AI_MODEL_FEATURE_COUNT);

  status = stai_network_run(ai_network_context, STAI_MODE_SYNC);
  if (status != STAI_SUCCESS)
  {
    return (int32_t)status;
  }

  memcpy(output, network_output, sizeof(float) * AI_MODEL_OUTPUT_COUNT);
  return 0;
}
