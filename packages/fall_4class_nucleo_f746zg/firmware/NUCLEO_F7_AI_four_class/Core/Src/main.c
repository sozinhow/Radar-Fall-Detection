/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "string.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "ai_model.h"
#include "radar_live.h"
#include <math.h>
#include <stdio.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
#if defined ( __ICCARM__ ) /*!< IAR Compiler */
#pragma location=0x2004c000
ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
#pragma location=0x2004c0a0
ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

#elif defined ( __CC_ARM )  /* MDK ARM Compiler */

__attribute__((at(0x2004c000))) ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
__attribute__((at(0x2004c0a0))) ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

#elif defined ( __GNUC__ ) /* GNU Compiler */

ETH_DMADescTypeDef DMARxDscrTab[ETH_RX_DESC_CNT] __attribute__((section(".RxDecripSection"))); /* Ethernet Rx DMA Descriptors */
ETH_DMADescTypeDef DMATxDscrTab[ETH_TX_DESC_CNT] __attribute__((section(".TxDecripSection")));   /* Ethernet Tx DMA Descriptors */
#endif

ETH_TxPacketConfig TxConfig;

ETH_HandleTypeDef heth;

I2C_HandleTypeDef hi2c1;

UART_HandleTypeDef huart3;
UART_HandleTypeDef huart6;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* USER CODE BEGIN PV */

#define RADAR_RX_RING_SIZE 2048U
#define RESULT_REPORT_PERIOD_MS 500U
#define RADAR_VERBOSE_DIAGNOSTICS 0U
#define USER_BUTTON_DEBOUNCE_MS 30U

#define RESULT_WALKING 0U
#define RESULT_STANDING 1U
#define RESULT_SITTING 2U
#define RESULT_FALL 3U
#define RESULT_TOO_CLOSE 4U
#define RESULT_TOO_FAR 5U
#define RESULT_STALE_DATA 6U
#define RESULT_EDGE_OF_FOV 7U

#define RADAR_TARGET_STALE_MS 250U
#define RADAR_ALERT_AZIMUTH_LIMIT_DEG 50.0f

static uint8_t radar_rx_byte;
static uint8_t radar_rx_ring[RADAR_RX_RING_SIZE];
static volatile uint16_t radar_rx_head;
static volatile uint16_t radar_rx_tail;
static volatile uint32_t radar_rx_dropped;
static volatile uint8_t radar_detection_enabled = 1U;
static uint8_t bridge_ready_message[] =
    "\r\nNUCLEO_F7_AI_four_class live four-class detector ready\r\n"
    "classes=[WALKING,STANDING,SITTING,FALL] input=60x7 sample_rate=20Hz z_mode=SENSOR_NATIVE\r\n"
    "target_selection=cluster_continuity zero_targets=RESET_WINDOW\r\n"
    "classification=argmax(logits)+softmax_probability guards=range+freshness+center_fov\r\n";
static char ai_status_message[320];
static float ai_self_test_output[AI_MODEL_OUTPUT_COUNT];
static float ai_live_output[AI_MODEL_OUTPUT_COUNT];
static uint8_t ai_ready;
static RadarLiveDiagnostics radar_diagnostics;
static uint8_t last_reported_result = 0xFFU;
static uint32_t last_result_report_ms;
static GPIO_PinState user_button_raw_state;
static GPIO_PinState user_button_stable_state;
static uint32_t user_button_change_ms;

static const char *const class_names[AI_MODEL_OUTPUT_COUNT] = {
    "WALKING", "STANDING", "SITTING", "FALL"};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_ETH_Init(void);
static void MX_I2C1_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_USART6_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

static int32_t FloatToMicroUnits(float value)
{
  if (value >= 0.0f)
  {
    return (int32_t)((value * 1000000.0f) + 0.5f);
  }
  return (int32_t)((value * 1000000.0f) - 0.5f);
}

static uint32_t ClassifyLogits(const float logits[AI_MODEL_OUTPUT_COUNT],
                               uint32_t probabilities_permille[AI_MODEL_OUTPUT_COUNT])
{
  float maximum = logits[0];
  float exponentials[AI_MODEL_OUTPUT_COUNT];
  float total = 0.0f;
  uint32_t predicted = 0U;
  uint32_t index;

  for (index = 1U; index < AI_MODEL_OUTPUT_COUNT; ++index)
  {
    if (logits[index] > maximum)
    {
      maximum = logits[index];
      predicted = index;
    }
  }

  for (index = 0U; index < AI_MODEL_OUTPUT_COUNT; ++index)
  {
    exponentials[index] = expf(logits[index] - maximum);
    total += exponentials[index];
  }

  if (!(total > 0.0f) || !isfinite(total))
  {
    for (index = 0U; index < AI_MODEL_OUTPUT_COUNT; ++index)
      probabilities_permille[index] = 0U;
    probabilities_permille[predicted] = 1000U;
    return predicted;
  }

  for (index = 0U; index < AI_MODEL_OUTPUT_COUNT; ++index)
  {
    probabilities_permille[index] =
        (uint32_t)((1000.0f * exponentials[index] / total) + 0.5f);
  }
  return predicted;
}

static void RadarDetection_ReportState(const char *reason)
{
  int message_length = snprintf(ai_status_message, sizeof(ai_status_message),
      "DETECTION=%s reason=%s window=%lu\r\n",
      (radar_detection_enabled != 0U) ? "RUNNING" : "STOPPED", reason,
      (unsigned long)RadarLive_GetWindowCount());
  if (message_length > 0)
    HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                      (uint16_t)message_length, 1000U);
}

static void RadarDetection_ResetPipeline(void)
{
  HAL_NVIC_DisableIRQ(USART6_IRQn);
  radar_rx_head = 0U;
  radar_rx_tail = 0U;
  (void)HAL_UART_AbortReceive(&huart6);
  RadarLive_Init();
  last_reported_result = 0xFFU;
  last_result_report_ms = 0U;
  __HAL_UART_CLEAR_OREFLAG(&huart6);
  (void)HAL_UART_Receive_IT(&huart6, &radar_rx_byte, 1U);
  HAL_NVIC_EnableIRQ(USART6_IRQn);
}

void RadarDetection_Start(void)
{
  if (radar_detection_enabled == 0U)
  {
    RadarDetection_ResetPipeline();
    radar_detection_enabled = 1U;
    HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_SET);
    HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
    RadarDetection_ReportState("START");
  }
}

void RadarDetection_Stop(void)
{
  if (radar_detection_enabled != 0U)
  {
    radar_detection_enabled = 0U;
    RadarDetection_ResetPipeline();
    HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_SET);
    RadarDetection_ReportState("STOP");
  }
}

void RadarDetection_Restart(void)
{
  radar_detection_enabled = 0U;
  RadarDetection_ResetPipeline();
  radar_detection_enabled = 1U;
  HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
  RadarDetection_ReportState("RESTART");
}

uint8_t RadarDetection_IsEnabled(void)
{
  return radar_detection_enabled;
}

static void ProcessDetectionCommand(void)
{
  uint8_t command;
  if (HAL_UART_Receive(&huart3, &command, 1U, 0U) != HAL_OK)
    return;
  if ((command == 'S') || (command == 's')) RadarDetection_Stop();
  else if ((command == 'G') || (command == 'g')) RadarDetection_Start();
  else if ((command == 'R') || (command == 'r')) RadarDetection_Restart();
  else if ((command == 'T') || (command == 't'))
  {
    if (RadarDetection_IsEnabled() != 0U) RadarDetection_Stop();
    else RadarDetection_Start();
  }
  else if (command == '?') RadarDetection_ReportState("STATUS");
}

static void ProcessUserButton(void)
{
  GPIO_PinState current_state = HAL_GPIO_ReadPin(USER_Btn_GPIO_Port, USER_Btn_Pin);
  uint32_t now_ms = HAL_GetTick();
  if (current_state != user_button_raw_state)
  {
    user_button_raw_state = current_state;
    user_button_change_ms = now_ms;
  }
  if ((current_state != user_button_stable_state) &&
      ((now_ms - user_button_change_ms) >= USER_BUTTON_DEBOUNCE_MS))
  {
    user_button_stable_state = current_state;
    if (current_state == GPIO_PIN_SET)
    {
      if (RadarDetection_IsEnabled() != 0U) RadarDetection_Stop();
      else RadarDetection_Start();
    }
  }
}
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART6)
  {
    uint16_t next_head = (uint16_t)((radar_rx_head + 1U) % RADAR_RX_RING_SIZE);
    if ((radar_detection_enabled != 0U) && (next_head != radar_rx_tail))
    {
      radar_rx_ring[radar_rx_head] = radar_rx_byte;
      radar_rx_head = next_head;
    }
    else
    {
      ++radar_rx_dropped;
    }
    (void)HAL_UART_Receive_IT(&huart6, &radar_rx_byte, 1U);
  }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART6)
  {
    __HAL_UART_CLEAR_OREFLAG(huart);
    (void)HAL_UART_Receive_IT(&huart6, &radar_rx_byte, 1U);
  }
}

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_ETH_Init();
  MX_I2C1_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  MX_USART6_UART_Init();
  /* USER CODE BEGIN 2 */

  HAL_UART_Transmit(&huart3, bridge_ready_message,
                    sizeof(bridge_ready_message) - 1U, 1000U);

  {
    int32_t ai_status = AI_Model_Init();
    if (ai_status == 0)
    {
      ai_status = AI_Model_RunSelfTest(ai_self_test_output);
    }

    if (ai_status == 0)
    {
      ai_ready = 1U;
      uint32_t self_test_probabilities[AI_MODEL_OUTPUT_COUNT];
      uint32_t predicted_class =
          ClassifyLogits(ai_self_test_output, self_test_probabilities);
      int message_length = snprintf(
          ai_status_message, sizeof(ai_status_message),
          "AI SELFTEST OK input=60x7 logits_x1e6=[%ld,%ld,%ld,%ld] class=%s probs_permille=[%lu,%lu,%lu,%lu]\r\n",
          (long)FloatToMicroUnits(ai_self_test_output[0]),
          (long)FloatToMicroUnits(ai_self_test_output[1]),
          (long)FloatToMicroUnits(ai_self_test_output[2]),
          (long)FloatToMicroUnits(ai_self_test_output[3]),
          class_names[predicted_class],
          (unsigned long)self_test_probabilities[0],
          (unsigned long)self_test_probabilities[1],
          (unsigned long)self_test_probabilities[2],
          (unsigned long)self_test_probabilities[3]);
      if (message_length > 0)
      {
        HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                          (uint16_t)message_length, 1000U);
      }
    }
    else
    {
      int message_length = snprintf(
          ai_status_message, sizeof(ai_status_message),
          "AI SELFTEST ERROR code=%ld\r\n", (long)ai_status);
      if (message_length > 0)
      {
        HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                          (uint16_t)message_length, 1000U);
      }
    }
  }

  RadarLive_Init();
  HAL_NVIC_SetPriority(USART6_IRQn, 5U, 0U);
  HAL_NVIC_EnableIRQ(USART6_IRQn);
  (void)HAL_UART_Receive_IT(&huart6, &radar_rx_byte, 1U);

  user_button_raw_state = HAL_GPIO_ReadPin(USER_Btn_GPIO_Port, USER_Btn_Pin);
  user_button_stable_state = user_button_raw_state;
  user_button_change_ms = HAL_GetTick();
  HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_SET);
  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
  RadarDetection_ReportState("BOOT");

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    RadarLiveEvent event;
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
    ProcessDetectionCommand();
    ProcessUserButton();

    if (radar_detection_enabled == 0U)
    {
      continue;
    }

    if (radar_rx_tail != radar_rx_head)
    {
      uint8_t received_byte = radar_rx_ring[radar_rx_tail];
      radar_rx_tail = (uint16_t)((radar_rx_tail + 1U) % RADAR_RX_RING_SIZE);
      (void)RadarLive_ProcessByte(received_byte, HAL_GetTick());
    }

    event = RadarLive_Tick(HAL_GetTick());
    if (event == RADAR_LIVE_TARGET_LOST)
      {
        int message_length;
        last_reported_result = RESULT_STALE_DATA;
        message_length = snprintf(
            ai_status_message, sizeof(ai_status_message),
            "RESULT=NO_TARGET action=WAIT_FOR_REACQUIRE window=RESET\r\n");
        if (message_length > 0)
        {
          HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                            (uint16_t)message_length, 1000U);
        }
      }
    else if (event == RADAR_LIVE_INFERENCE_READY && ai_ready != 0U)
      {
        int32_t ai_status = AI_Model_Run(RadarLive_GetModelInput(), ai_live_output);
        if (ai_status == 0)
        {
          uint8_t effective_result;
          uint32_t now_ms = HAL_GetTick();
          uint32_t class_probabilities[AI_MODEL_OUTPUT_COUNT];
          RadarLive_GetDiagnostics(&radar_diagnostics);
          if (radar_diagnostics.latest_target_age_ms > RADAR_TARGET_STALE_MS)
          {
            effective_result = RESULT_STALE_DATA;
          }
          else if (radar_diagnostics.range_quality == RADAR_RANGE_TOO_CLOSE)
          {
            effective_result = RESULT_TOO_CLOSE;
          }
          else if (radar_diagnostics.range_quality == RADAR_RANGE_TOO_FAR)
          {
            effective_result = RESULT_TOO_FAR;
          }
          else if (fabsf(radar_diagnostics.latest_azimuth_deg) >
                   RADAR_ALERT_AZIMUTH_LIMIT_DEG)
          {
            effective_result = RESULT_EDGE_OF_FOV;
          }
          else
          {
            effective_result = (uint8_t)ClassifyLogits(
                ai_live_output, class_probabilities);
          }

          if ((effective_result != last_reported_result) ||
              ((now_ms - last_result_report_ms) >= RESULT_REPORT_PERIOD_MS))
          {
            int message_length;
            if (effective_result == RESULT_STALE_DATA)
            {
              message_length = snprintf(
                  ai_status_message, sizeof(ai_status_message),
                  "RESULT=NO_FRESH_TARGET age_ms=%lu action=CHECK_RADAR\r\n",
                  (unsigned long)radar_diagnostics.latest_target_age_ms);
            }
            else if (effective_result == RESULT_TOO_CLOSE)
            {
              message_length = snprintf(
                  ai_status_message, sizeof(ai_status_message),
                  "RESULT=TOO_CLOSE distance_cm=%ld minimum_cm=50\r\n",
                  (long)(radar_diagnostics.latest_range * 100.0f));
            }
            else if (effective_result == RESULT_TOO_FAR)
            {
              message_length = snprintf(
                  ai_status_message, sizeof(ai_status_message),
                  "RESULT=TOO_FAR distance_cm=%ld action=MOVE_CLOSER\r\n",
                  (long)(radar_diagnostics.latest_range * 100.0f));
            }
            else if (effective_result == RESULT_EDGE_OF_FOV)
            {
              message_length = snprintf(
                  ai_status_message, sizeof(ai_status_message),
                  "RESULT=EDGE_OF_FOV azimuth_deg=%ld action=RETURN_TO_CENTER\r\n",
                  (long)radar_diagnostics.latest_azimuth_deg);
            }
            else
            {
              const char *result_name = class_names[effective_result];
              message_length = snprintf(
                  ai_status_message, sizeof(ai_status_message),
                  "RESULT=%s class=%u probs_permille=[%lu,%lu,%lu,%lu] targets=%lu cluster=%ld range_cm=%ld az_deg=%ld xyz_cm=[%ld,%ld,%ld]\r\n",
                  result_name, (unsigned int)effective_result,
                  (unsigned long)class_probabilities[0],
                  (unsigned long)class_probabilities[1],
                  (unsigned long)class_probabilities[2],
                  (unsigned long)class_probabilities[3],
                  (unsigned long)radar_diagnostics.latest_target_count,
                  (long)radar_diagnostics.latest_cluster_id,
                  (long)(radar_diagnostics.latest_range * 100.0f),
                  (long)radar_diagnostics.latest_azimuth_deg,
                  (long)(radar_diagnostics.latest_x * 100.0f),
                  (long)(radar_diagnostics.latest_y * 100.0f),
                  (long)(radar_diagnostics.latest_z * 100.0f));
            }

            last_reported_result = effective_result;
            last_result_report_ms = now_ms;
            if (message_length > 0)
            {
              HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                                (uint16_t)message_length, 1000U);
            }
          }

#if RADAR_VERBOSE_DIAGNOSTICS
          {
            int message_length = snprintf(
                ai_status_message, sizeof(ai_status_message),
                "DEBUG range_mm=%ld sensor_z_mm=%ld model_z_mm=%ld dt_ms=%lu age_ms=%lu frames=%lu invalid=%lu dropped=%lu\r\n",
                (long)(radar_diagnostics.latest_range * 1000.0f),
                (long)(radar_diagnostics.latest_sensor_z * 1000.0f),
                (long)(radar_diagnostics.latest_z * 1000.0f),
                (unsigned long)radar_diagnostics.last_sample_interval_ms,
                (unsigned long)radar_diagnostics.latest_target_age_ms,
                (unsigned long)RadarLive_GetValidFrameCount(),
                (unsigned long)RadarLive_GetInvalidFrameCount(),
                (unsigned long)radar_rx_dropped);
            if (message_length > 0)
            {
              HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                                (uint16_t)message_length, 1000U);
            }
          }
#endif
        }
        else
        {
          int message_length = snprintf(
              ai_status_message, sizeof(ai_status_message),
              "AI LIVE ERROR code=%ld\r\n", (long)ai_status);
          if (message_length > 0)
          {
            HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                              (uint16_t)message_length, 1000U);
          }
        }
      }
    else if (event == RADAR_LIVE_SAMPLE_ADDED &&
               (RadarLive_GetWindowCount() % 10U) == 0U &&
               RadarLive_GetWindowCount() < AI_MODEL_FRAME_COUNT)
      {
        RadarLive_GetDiagnostics(&radar_diagnostics);
        int message_length = snprintf(
            ai_status_message, sizeof(ai_status_message),
            "WARMUP %lu/60\r\n",
            (unsigned long)RadarLive_GetWindowCount());
        if (message_length > 0)
        {
          HAL_UART_Transmit(&huart3, (uint8_t *)ai_status_message,
                            (uint16_t)message_length, 1000U);
        }
      }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure LSE Drive Capability
  */
  HAL_PWR_EnableBkUpAccess();

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 72;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 3;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ETH Initialization Function
  * @param None
  * @retval None
  */
static void MX_ETH_Init(void)
{

  /* USER CODE BEGIN ETH_Init 0 */

  /* USER CODE END ETH_Init 0 */

   static uint8_t MACAddr[6];

  /* USER CODE BEGIN ETH_Init 1 */

  /* USER CODE END ETH_Init 1 */
  heth.Instance = ETH;
  MACAddr[0] = 0x00;
  MACAddr[1] = 0x80;
  MACAddr[2] = 0xE1;
  MACAddr[3] = 0x00;
  MACAddr[4] = 0x00;
  MACAddr[5] = 0x00;
  heth.Init.MACAddr = &MACAddr[0];
  heth.Init.MediaInterface = HAL_ETH_RMII_MODE;
  heth.Init.TxDesc = DMATxDscrTab;
  heth.Init.RxDesc = DMARxDscrTab;
  heth.Init.RxBuffLen = 1524;

  /* USER CODE BEGIN MACADDRESS */

  /* USER CODE END MACADDRESS */

  if (HAL_ETH_Init(&heth) != HAL_OK)
  {
    Error_Handler();
  }

  memset(&TxConfig, 0 , sizeof(ETH_TxPacketConfig));
  TxConfig.Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD;
  TxConfig.ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC;
  TxConfig.CRCPadCtrl = ETH_CRC_PAD_INSERT;
  /* USER CODE BEGIN ETH_Init 2 */

  /* USER CODE END ETH_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x00808CD2;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief USART6 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART6_UART_Init(void)
{

  /* USER CODE BEGIN USART6_Init 0 */

  /* USER CODE END USART6_Init 0 */

  /* USER CODE BEGIN USART6_Init 1 */

  /* USER CODE END USART6_Init 1 */
  huart6.Instance = USART6;
  huart6.Init.BaudRate = 115200;
  huart6.Init.WordLength = UART_WORDLENGTH_8B;
  huart6.Init.StopBits = UART_STOPBITS_1;
  huart6.Init.Parity = UART_PARITY_NONE;
  huart6.Init.Mode = UART_MODE_TX_RX;
  huart6.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart6.Init.OverSampling = UART_OVERSAMPLING_16;
  huart6.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart6.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart6) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART6_Init 2 */

  /* USER CODE END USART6_Init 2 */

}

/**
  * @brief USB_OTG_FS Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_OTG_FS_PCD_Init(void)
{

  /* USER CODE BEGIN USB_OTG_FS_Init 0 */

  /* USER CODE END USB_OTG_FS_Init 0 */

  /* USER CODE BEGIN USB_OTG_FS_Init 1 */

  /* USER CODE END USB_OTG_FS_Init 1 */
  hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
  hpcd_USB_OTG_FS.Init.dev_endpoints = 6;
  hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
  hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_OTG_FS_Init 2 */

  /* USER CODE END USB_OTG_FS_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin|LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB_PowerSwitchOn_GPIO_Port, USB_PowerSwitchOn_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : USER_Btn_Pin */
  GPIO_InitStruct.Pin = USER_Btn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USER_Btn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin LD2_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|LD3_Pin|LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = USB_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OverCurrent_Pin */
  GPIO_InitStruct.Pin = USB_OverCurrent_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_OverCurrent_GPIO_Port, &GPIO_InitStruct);

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
