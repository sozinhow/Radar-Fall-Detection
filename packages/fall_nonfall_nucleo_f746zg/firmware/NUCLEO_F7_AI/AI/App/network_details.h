/**
  ******************************************************************************
  * @file    network.h
  * @date    2026-08-18T07:18:01+0000
  * @brief   ST.AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */
#ifndef STAI_NETWORK_DETAILS_H
#define STAI_NETWORK_DETAILS_H

#include "stai.h"
#include "layers.h"

const stai_network_details g_network_details = {
  .tensors = (const stai_tensor[21]) {
   { .size_bytes = 1680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 7}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "radar_input_output" },
   { .size_bytes = 7680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_0_Conv_output_0_output" },
   { .size_bytes = 7680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 60, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_2_Relu_output_0_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 30, 32}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_3_MaxPool_output_0_output" },
   { .size_bytes = 7680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 30, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_4_Conv_output_0_output" },
   { .size_bytes = 7680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 30, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_6_Relu_output_0_output" },
   { .size_bytes = 3840, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 15, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_7_MaxPool_output_0_output" },
   { .size_bytes = 5760, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 15, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_8_Conv_output_0_output" },
   { .size_bytes = 5760, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 15, 96}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_10_Relu_output_0_output" },
   { .size_bytes = 7680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 15, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_11_Conv_output_0_output" },
   { .size_bytes = 7680, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 15, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_features_features_13_Relu_output_0_output" },
   { .size_bytes = 512, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {3, (const int32_t[3]){1, 1, 128}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_pool_GlobalAveragePool_output_0_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_classifier_classifier_1_Gemm_output_0_output" },
   { .size_bytes = 256, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 64}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_classifier_classifier_2_Relu_output_0_output" },
   { .size_bytes = 16, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 4}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_base_model_classifier_classifier_3_Gemm_output_0_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Slice_1_output_0_output" },
   { .size_bytes = 12, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 3}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_Slice_output_0_output" },
   { .size_bytes = 12, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 3}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_ReduceLogSumExp_output_0_exp_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_ReduceLogSumExp_output_0_reduce_output" },
   { .size_bytes = 4, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 1}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "_ReduceLogSumExp_output_0_output" },
   { .size_bytes = 8, .flags = (STAI_FLAG_HAS_BATCH|STAI_FLAG_CHANNEL_LAST), .format = STAI_FORMAT_FLOAT32, .shape = {2, (const int32_t[2]){1, 2}}, .scale = {0, NULL}, .zeropoint = {0, NULL}, .name = "binary_logits_output" }
  },
  .nodes = (const stai_node_details[20]){
    {.id = 2, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){0}}, .output_tensors = {1, (const int32_t[1]){1}} }, /* _base_model_features_features_0_Conv_output_0 */
    {.id = 3, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){1}}, .output_tensors = {1, (const int32_t[1]){2}} }, /* _base_model_features_features_2_Relu_output_0 */
    {.id = 4, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){2}}, .output_tensors = {1, (const int32_t[1]){3}} }, /* _base_model_features_features_3_MaxPool_output_0 */
    {.id = 5, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){3}}, .output_tensors = {1, (const int32_t[1]){4}} }, /* _base_model_features_features_4_Conv_output_0 */
    {.id = 6, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){4}}, .output_tensors = {1, (const int32_t[1]){5}} }, /* _base_model_features_features_6_Relu_output_0 */
    {.id = 7, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){5}}, .output_tensors = {1, (const int32_t[1]){6}} }, /* _base_model_features_features_7_MaxPool_output_0 */
    {.id = 8, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){6}}, .output_tensors = {1, (const int32_t[1]){7}} }, /* _base_model_features_features_8_Conv_output_0 */
    {.id = 9, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){7}}, .output_tensors = {1, (const int32_t[1]){8}} }, /* _base_model_features_features_10_Relu_output_0 */
    {.id = 10, .type = AI_LAYER_CONV2D_TYPE, .input_tensors = {1, (const int32_t[1]){8}}, .output_tensors = {1, (const int32_t[1]){9}} }, /* _base_model_features_features_11_Conv_output_0 */
    {.id = 11, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){9}}, .output_tensors = {1, (const int32_t[1]){10}} }, /* _base_model_features_features_13_Relu_output_0 */
    {.id = 12, .type = AI_LAYER_POOL_TYPE, .input_tensors = {1, (const int32_t[1]){10}}, .output_tensors = {1, (const int32_t[1]){11}} }, /* _base_model_pool_GlobalAveragePool_output_0 */
    {.id = 15, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){11}}, .output_tensors = {1, (const int32_t[1]){12}} }, /* _base_model_classifier_classifier_1_Gemm_output_0 */
    {.id = 16, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){12}}, .output_tensors = {1, (const int32_t[1]){13}} }, /* _base_model_classifier_classifier_2_Relu_output_0 */
    {.id = 17, .type = AI_LAYER_DENSE_TYPE, .input_tensors = {1, (const int32_t[1]){13}}, .output_tensors = {1, (const int32_t[1]){14}} }, /* _base_model_classifier_classifier_3_Gemm_output_0 */
    {.id = 28, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){15}} }, /* _Slice_1_output_0 */
    {.id = 22, .type = AI_LAYER_SLICE_TYPE, .input_tensors = {1, (const int32_t[1]){14}}, .output_tensors = {1, (const int32_t[1]){16}} }, /* _Slice_output_0 */
    {.id = 23, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){16}}, .output_tensors = {1, (const int32_t[1]){17}} }, /* _ReduceLogSumExp_output_0_exp */
    {.id = 24, .type = AI_LAYER_REDUCE_TYPE, .input_tensors = {1, (const int32_t[1]){17}}, .output_tensors = {1, (const int32_t[1]){18}} }, /* _ReduceLogSumExp_output_0_reduce */
    {.id = 25, .type = AI_LAYER_NL_TYPE, .input_tensors = {1, (const int32_t[1]){18}}, .output_tensors = {1, (const int32_t[1]){19}} }, /* _ReduceLogSumExp_output_0 */
    {.id = 29, .type = AI_LAYER_CONCAT_TYPE, .input_tensors = {2, (const int32_t[2]){19, 15}}, .output_tensors = {1, (const int32_t[1]){20}} } /* binary_logits */
  },
  .n_nodes = 20
};
#endif

